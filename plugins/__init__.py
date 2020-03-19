"""
    Copyright 2016 Inmanta

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Contact: code@inmanta.com
"""

import base64
import hashlib
import importlib
import logging
import os
import random
import re
import time
from collections import defaultdict
from copy import copy
from itertools import chain
from operator import attrgetter

import jinja2
import pydantic
from inmanta import resources
from inmanta.ast import NotFoundException, OptionalValueException, RuntimeException
from inmanta.config import Config
from inmanta.execute.proxy import DynamicProxy, UnknownException
from inmanta.execute.util import NoneValue, Unknown
from inmanta.export import dependency_manager, unknown_parameters
from inmanta.module import Project
from inmanta.plugins import Context, plugin
from jinja2 import Environment, FileSystemLoader, PrefixLoader
from jinja2.exceptions import UndefinedError
from jinja2.runtime import Undefined


@plugin
def unique_file(
    prefix: "string", seed: "string", suffix: "string", length: "number" = 20
) -> "string":
    return prefix + hashlib.md5(seed.encode("utf-8")).hexdigest() + suffix


tcache = {}

engine_cache = None


class JinjaDynamicProxy(DynamicProxy):
    def __init__(self, instance):
        super(JinjaDynamicProxy, self).__init__(instance)

    @classmethod
    def return_value(cls, value):
        if value is None:
            return None

        if isinstance(value, NoneValue):
            return None

        if isinstance(value, Unknown):
            raise UnknownException(value)

        if isinstance(value, (str, tuple, int, float, bool)):
            return copy(value)

        if isinstance(value, DynamicProxy):
            return value

        if hasattr(value, "__len__"):
            return SequenceProxy(value)

        if hasattr(value, "__call__"):
            return CallProxy(value)

        return cls(value)

    def __getattr__(self, attribute):
        instance = self._get_instance()
        if hasattr(instance, "get_attribute"):
            try:
                value = instance.get_attribute(attribute).get_value()
                return JinjaDynamicProxy.return_value(value)
            except (OptionalValueException, NotFoundException):
                return Undefined(
                    "variable %s not set on %s" % (attribute, instance),
                    instance,
                    attribute,
                )
        else:
            # A native python object such as a dict
            return getattr(instance, attribute)


class SequenceProxy(JinjaDynamicProxy):
    def __init__(self, iterator):
        JinjaDynamicProxy.__init__(self, iterator)

    def __getitem__(self, key):
        instance = self._get_instance()
        if isinstance(key, str):
            raise RuntimeException(
                self,
                "can not get a attribute %s, %s is a list"
                % (key, self._get_instance()),
            )

        return JinjaDynamicProxy.return_value(instance[key])

    def __len__(self):
        return len(self._get_instance())

    def __iter__(self):
        instance = self._get_instance()

        return IteratorProxy(instance.__iter__())

    def items(self):
        return self._get_instance().items()


class CallProxy(JinjaDynamicProxy):
    """
        Proxy a value that implements a __call__ function
    """

    def __init__(self, instance):
        JinjaDynamicProxy.__init__(self, instance)

    def __call__(self, *args, **kwargs):
        instance = self._get_instance()

        return instance(*args, **kwargs)


class IteratorProxy(JinjaDynamicProxy):
    """
        Proxy an iterator call
    """

    def __init__(self, iterator):
        JinjaDynamicProxy.__init__(self, iterator)

    def __iter__(self):
        return self

    def __next__(self):
        i = self._get_instance()
        return JinjaDynamicProxy.return_value(next(i))


class ResolverContext(jinja2.runtime.Context):
    def resolve(self, key):
        resolver = self.parent["{{resolver"]
        try:
            raw = resolver.lookup(key)
            return JinjaDynamicProxy.return_value(raw.get_value())
        except NotFoundException:
            return super(ResolverContext, self).resolve(key)
        except OptionalValueException as e:
            return self.environment.undefined(
                "variable %s not set on %s" % (resolver, key), resolver, key, e
            )


def _get_template_engine(ctx):
    """
        Initialize the template engine environment
    """
    global engine_cache
    if engine_cache is not None:
        return engine_cache

    loader_map = {}
    loader_map[""] = FileSystemLoader(
        os.path.join(Project.get().project_path, "templates")
    )
    for name, module in Project.get().modules.items():
        template_dir = os.path.join(module._path, "templates")
        if os.path.isdir(template_dir):
            loader_map[name] = FileSystemLoader(template_dir)

    # init the environment
    env = Environment(loader=PrefixLoader(loader_map), undefined=jinja2.StrictUndefined)
    env.context_class = ResolverContext

    # register all plugins as filters
    for name, cls in ctx.get_compiler().get_plugins().items():

        def curywrapper(func):
            def safewrapper(*args):
                return JinjaDynamicProxy.return_value(func(*args))

            return safewrapper

        env.filters[name.replace("::", ".")] = curywrapper(cls)

    engine_cache = env
    return env


def _extend_path(ctx: Context, path: str):
    current_module_prefix = "." + os.path.sep
    if path.startswith(current_module_prefix):
        module_and_submodule_name_parts = ctx.owner.namespace.get_full_name().split(
            "::"
        )
        module_name = module_and_submodule_name_parts[0]
        if module_name in Project.get().modules.keys():
            return os.path.join(module_name, path[len(current_module_prefix) :])
        else:
            raise Exception(
                f"Unable to determine current module for path {path}, called from {ctx.owner.namespace.get_full_name()}"
            )
    return path


@plugin("template")
def template(ctx: Context, path: "string"):
    """
        Execute the template in path in the current context. This function will
        generate a new statement that has dependencies on the used variables.
    """
    jinja_env = _get_template_engine(ctx)
    template_path = _extend_path(ctx, path)
    if template_path in tcache:
        template = tcache[template_path]
    else:
        template = jinja_env.get_template(template_path)
        tcache[template_path] = template

    resolver = ctx.get_resolver()

    try:
        out = template.render({"{{resolver": resolver})
        return out
    except UndefinedError as e:
        raise NotFoundException(ctx.owner, None, e.message)


@dependency_manager
def dir_before_file(model, resources):
    """
        If a file is defined on a host, then make the file depend on its parent directory
    """
    # loop over all resources to find files and dirs
    per_host = defaultdict(list)
    per_host_dirs = defaultdict(list)
    for _id, resource in resources.items():
        if (
            resource.id.get_entity_type() == "std::File"
            or resource.id.get_entity_type() == "std::Directory"
        ):
            per_host[resource.model.host].append(resource)

        if resource.id.get_entity_type() == "std::Directory":
            per_host_dirs[resource.model.host].append(resource)

    # now add deps per host
    for host, files in per_host.items():
        for hfile in files:
            for pdir in per_host_dirs[host]:
                if (
                    hfile.path != pdir.path
                    and hfile.path[: len(pdir.path)] == pdir.path
                ):
                    # Make the File resource require the directory
                    hfile.requires.add(pdir)


def get_passwords(pw_file):
    records = {}
    if os.path.exists(pw_file):
        with open(pw_file, "r") as fd:

            for line in fd.readlines():
                line = line.strip()
                if len(line) > 2:
                    i = line.index("=")

                    try:
                        records[line[:i].strip()] = line[i + 1 :].strip()
                    except ValueError:
                        pass

    return records


def save_passwords(pw_file, records):
    with open(pw_file, "w+") as fd:
        for key, value in records.items():
            fd.write("%s=%s\n" % (key, value))


@plugin
def generate_password(
    context: Context, pw_id: "string", length: "number" = 20
) -> "string":
    """
    Generate a new random password and store it in the data directory of the
    project. On next invocations the stored password will be used.

    :param pw_id: The id of the password to identify it.
    :param length: The length of the password, default length is 20
    """
    data_dir = context.get_data_dir()
    pw_file = os.path.join(data_dir, "passwordfile.txt")

    if "=" in pw_id:
        raise Exception("The password id cannot contain =")

    records = get_passwords(pw_file)

    if pw_id in records:
        return records[pw_id]

    rnd = random.SystemRandom()
    pw = ""
    while len(pw) < length:
        x = chr(rnd.randint(33, 126))
        if re.match("[A-Za-z0-9]", x) is not None:
            pw += x

    # store the new value
    records[pw_id] = pw
    save_passwords(pw_file, records)

    return pw


@plugin
def password(context: Context, pw_id: "string") -> "string":
    """
        Retrieve the given password from a password file. It raises an exception when a password is not found

        :param pw_id: The id of the password to identify it.
    """
    data_dir = context.get_data_dir()
    pw_file = os.path.join(data_dir, "passwordfile.txt")

    if "=" in pw_id:
        raise Exception("The password id cannot contain =")

    records = get_passwords(pw_file)

    if pw_id in records:
        return records[pw_id]

    else:
        raise Exception("Password %s does not exist in file %s" % (pw_id, pw_file))


@plugin("print")
def printf(message: "any"):
    """
        Print the given message to stdout
    """
    print(message)


@plugin
def replace(string: "string", old: "string", new: "string") -> "string":
    return string.replace(old, new)


@plugin
def equals(arg1: "any", arg2: "any", desc: "string" = None):
    """
        Compare arg1 and arg2
    """
    if arg1 != arg2:
        if desc is not None:
            raise AssertionError("%s != %s: %s" % (arg1, arg2, desc))
        else:
            raise AssertionError("%s != %s" % (arg1, arg2))


@plugin("assert")
def assert_function(expression: "bool", message: "string" = ""):
    """
        Raise assertion error if expression is false
    """
    if not expression:
        raise AssertionError("Assertion error: " + message)


@plugin
def select(objects: "list", attr: "string") -> "list":
    """
        Return a list with the select attributes
    """
    return [getattr(item, attr) for item in objects]


@plugin
def item(objects: "list", index: "number") -> "list":
    """
        Return a list that selects the item at index from each of the sublists
    """
    r = []
    for obj in objects:
        r.append(obj[index])

    return r


@plugin
def key_sort(items: "list", key: "any") -> "list":
    """
        Sort an array of object on key
    """
    if isinstance(key, tuple):
        return sorted(items, key=attrgetter(*key))

    return sorted(items, key=attrgetter(key))


@plugin
def timestamp(dummy: "any" = None) -> "number":
    """
        Return an integer with the current unix timestamp

        :param any: A dummy argument to be able to use this function as a filter
    """
    return int(time.time())


@plugin
def capitalize(string: "string") -> "string":
    """
        Capitalize the given string
    """
    return string.capitalize()


@plugin
def type(obj: "any") -> "any":
    value = obj.value
    return value.type().__definition__


@plugin
def sequence(i: "number", start: "number" = 0, offset: "number" = 0) -> "list":
    """
        Return a sequence of i numbers, starting from zero or start if supplied.
    """
    return list(range(start, int(i) + start - offset))


@plugin
def inlineif(conditional: "bool", a: "any", b: "any") -> "any":
    """
        An inline if
    """
    if conditional:
        return a
    return b


@plugin
def at(objects: "list", index: "number") -> "any":
    """
        Get the item at index
    """
    return objects[int(index)]


@plugin
def attr(obj: "any", attr: "string") -> "any":
    return getattr(obj, attr)


@plugin
def isset(value: "any") -> "bool":
    """
        Returns true if a value has been set
    """
    return value is not None


@plugin
def objid(value: "any") -> "string":
    return str(
        (
            value._get_instance(),
            str(id(value._get_instance())),
            value._get_instance().__class__,
        )
    )


@plugin
def count(item_list: "list") -> "number":
    """
        Returns the number of elements in this list
    """
    return len(item_list)


@plugin
def unique(item_list: "list") -> "bool":
    """
        Returns true if all items in this sequence are unique
    """
    seen = set()
    for item in item_list:
        if item in seen:
            return False
        seen.add(item)

    return True


@plugin
def flatten(item_list: "list") -> "list":
    """
        Flatten this list
    """
    return list(chain.from_iterable(item_list))


@plugin
def split(string_list: "string", delim: "string") -> "list":
    """
        Split the given string into a list

        :param string_list: The list to split into parts
        :param delim: The delimeter to split the text by
    """
    return string_list.split(delim)


def determine_path(ctx, module_dir, path):
    """
        Determine the real path based on the given path
    """
    path = _extend_path(ctx, path)
    parts = path.split(os.path.sep)

    modules = Project.get().modules

    if parts[0] == "":
        module_path = Project.get().project_path
    elif parts[0] not in modules:
        raise Exception("Module %s does not exist for path %s" % (parts[0], path))
    else:
        module_path = modules[parts[0]]._path

    return os.path.join(module_path, module_dir, os.path.sep.join(parts[1:]))


def get_file_content(ctx, module_dir, path):
    """
        Get the contents of a file
    """
    filename = determine_path(ctx, module_dir, path)

    if filename is None:
        raise Exception("%s does not exist" % path)

    if not os.path.isfile(filename):
        raise Exception("%s isn't a valid file (%s)" % (path, filename))

    file_fd = open(filename, "r")
    if file_fd is None:
        raise Exception("Unable to open file %s" % filename)

    content = file_fd.read()
    file_fd.close()

    return content


@plugin
def source(ctx: Context, path: "string") -> "string":
    """
        Return the textual contents of the given file
    """
    return get_file_content(ctx, "files", path)


class FileMarker(str):
    """
        Marker class to indicate that this string is actually a reference to a file on disk.

        This mechanism is backward compatible with the old in-band mechanism.

        To pass file references from other modules, you can copy paste this class into your own module.
        The matching in the file handler is:

            if "FileMarker" in content.__class__.__name__

    """

    def __new__(cls, filename):
        obj = str.__new__(cls, "imp-module-source:file://" + filename)
        obj.filename = filename
        return obj


@plugin
def file(ctx: Context, path: "string") -> "string":
    """
        Return the textual contents of the given file
    """
    filename = determine_path(ctx, "files", path)

    if filename is None:
        raise Exception("%s does not exist" % path)

    if not os.path.isfile(filename):
        raise Exception("%s isn't a valid file" % path)

    return FileMarker(os.path.abspath(filename))


@plugin
def familyof(member: "std::OS", family: "string") -> "bool":
    """
        Determine if member is a member of the given operating system family
    """
    if member.name == family:
        return True

    parent = member
    try:
        while parent.family is not None:
            if parent.name == family:
                return True

            parent = parent.family
    except OptionalValueException:
        pass

    return False


fact_cache = {}


@plugin
def getfact(
    context: Context, resource: "any", fact_name: "string", default_value: "any" = None
) -> "any":
    """
        Retrieve a fact of the given resource
    """
    global fact_cache

    resource_id = resources.to_id(resource)
    if resource_id is None:
        raise Exception("Facts can only be retreived from resources.")

    # Special case for unit testing and mocking
    if hasattr(context.compiler, "refs") and "facts" in context.compiler.refs:
        if (
            resource_id in context.compiler.refs["facts"]
            and fact_name in context.compiler.refs["facts"][resource_id]
        ):
            return context.compiler.refs["facts"][resource_id][fact_name]

        fact_value = Unknown(source=resource)
        unknown_parameters.append(
            {"resource": resource_id, "parameter": fact_name, "source": "fact"}
        )

        if default_value is not None:
            return default_value
        return fact_value
    # End special case

    try:
        client = context.get_client()

        env = Config.get("config", "environment", None)
        if env is None:
            raise Exception(
                "The environment of this model should be configured in config>environment"
            )

        # load cache
        if not fact_cache:

            def call():
                return client.list_params(tid=env,)

            result = context.run_sync(call)
            if result.code == 200:
                fact_values = result.result["parameters"]
                for fact_value in fact_values:
                    fact_cache.setdefault(fact_value["resource_id"], {})[
                        fact_value["name"]
                    ] = fact_value["value"]

        # attempt cache hit
        if resource_id in fact_cache:
            if fact_name in fact_cache[resource_id]:
                return fact_cache[resource_id][fact_name]

        fact_value = None

        def call():
            return client.get_param(tid=env, id=fact_name, resource_id=resource_id)

        result = context.run_sync(call)

        if result.code == 200:
            fact_value = result.result["parameter"]["value"]
        else:
            logging.getLogger(__name__).info(
                "Param %s of resource %s is unknown", fact_name, resource_id
            )
            fact_value = Unknown(source=resource)
            unknown_parameters.append(
                {"resource": resource_id, "parameter": fact_name, "source": "fact"}
            )

    except ConnectionRefusedError:
        logging.getLogger(__name__).warning(
            "Param %s of resource %s is unknown because connection to server was refused",
            fact_name,
            resource_id,
        )
        fact_value = Unknown(source=resource)
        unknown_parameters.append(
            {"resource": resource_id, "parameter": fact_name, "source": "fact"}
        )

    if isinstance(fact_value, Unknown) and default_value is not None:
        return default_value

    return fact_value


@plugin
def environment() -> "string":
    """
        Return the environment id
    """
    env = str(Config.get("config", "environment", None))

    if env is None:
        raise Exception(
            "The environment of this model should be configured in config>environment"
        )

    return str(env)


@plugin
def environment_name(ctx: Context) -> "string":
    """
        Return the name of the environment (as defined on the server)
    """
    env = environment()

    def call():
        return ctx.get_client().get_environment(id=env)

    result = ctx.run_sync(call)
    if result.code != 200:
        return Unknown(source=env)
    return result.result["environment"]["name"]


@plugin
def environment_server(ctx: Context) -> "string":
    """
        Return the address of the management server
    """
    client = ctx.get_client()
    server_url = client._transport_instance._get_client_config()
    match = re.search("^http[s]?://([^:]+):", server_url)
    if match is not None:
        return match.group(1)
    return Unknown(source=server_url)


@plugin
def is_set(obj: "any", attribute: "string") -> "bool":
    try:
        getattr(obj, attribute)
    except Exception:
        return False
    return True


@plugin
def server_ca() -> "string":
    filename = Config.get("compiler_rest_transport", "ssl_ca_cert_file", "")
    if filename == "":
        return ""
    if filename is None:
        raise Exception("%s does not exist" % filename)

    if not os.path.isfile(filename):
        raise Exception("%s isn't a valid file" % filename)

    file_fd = open(filename, "r")
    if file_fd is None:
        raise Exception("Unable to open file %s" % filename)

    content = file_fd.read()
    return content


@plugin
def server_ssl() -> "bool":
    return Config.get("compiler_rest_transport", "ssl", False)


@plugin
def server_token(context: Context, client_types: "string[]" = ["agent"]) -> "string":
    token = Config.get("compiler_rest_transport", "token", "")
    if token == "":
        return ""

    # Request a new token for this agent
    token = ""
    try:
        client = context.get_client()

        env = Config.get("config", "environment", None)
        if env is None:
            raise Exception(
                "The environment of this model should be configured in config>environment"
            )

        def call():
            return client.create_token(
                tid=env, client_types=list(client_types), idempotent=True
            )

        result = context.run_sync(call)

        if result.code == 200:
            token = result.result["token"]
        else:
            logging.getLogger(__name__).warning("Unable to get a new token")
            raise Exception("Unable to get a valid token")
    except ConnectionRefusedError:
        logging.getLogger(__name__).exception("Unable to get a new token")
        raise Exception("Unable to get a valid token")

    return token


@plugin
def server_port() -> "number":
    return Config.get("compiler_rest_transport", "port", 8888)


@plugin
def get_env(name: "string", default_value: "string" = None) -> "string":
    env = os.environ
    if name in env:
        return env[name]
    elif default_value is not None:
        return default_value
    else:
        return Unknown(source=name)


@plugin
def get_env_int(name: "string", default_value: "number" = None) -> "number":
    env = os.environ
    if name in env:
        return int(env[name])
    elif default_value is not None:
        return default_value
    else:
        return Unknown(source=name)


@plugin
def is_instance(ctx: Context, obj: "any", cls: "string") -> "bool":
    t = ctx.get_type(cls)
    try:
        t.validate(obj._get_instance())
    except RuntimeException:
        return False
    return True


@plugin
def length(value: "string") -> "number":
    """
        Return the length of the string
    """
    return len(value)


@plugin
def filter(values: "list", not_item: "std::Entity") -> "list":
    """
        Filter not_item from values
    """
    return [x for x in values if x != not_item]


@plugin
def dict_get(dct: "dict", key: "string") -> "string":
    """
        Get an element from the dict. Raises an exception when the key is not found in the dict
    """
    return dct[key]


@plugin
def contains(dct: "dict", key: "string") -> "bool":
    """
        Check if key exists in dct.
    """
    return key in dct


@plugin("getattr", allow_unknown=True)
def getattribute(
    entity: "std::Entity",
    attribute_name: "string",
    default_value: "any" = None,
    no_unknown: "bool" = True,
) -> "any":
    """
        Return the value of the given attribute. If the attribute does not exist, return the default value.

        :attr no_unknown: When this argument is set to true, this method will return the default value when the attribute
                          is unknown.
    """
    try:
        value = getattr(entity, attribute_name)
        if isinstance(value, Unknown) and no_unknown:
            return default_value
        return value
    except (NotFoundException, KeyError):
        return default_value


@plugin
def invert(value: "bool") -> "bool":
    """
        Invert a boolean value
    """
    return not value


@plugin
def to_number(value: "any") -> "number":
    """
        Convert a value to a number
    """
    return int(value)


@plugin
def list_files(ctx: Context, path: "string") -> "list":
    """
        List files in a directory
    """
    path = determine_path(ctx, "files", path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


@plugin(allow_unknown=True)
def is_unknown(value: "any") -> "bool":
    return isinstance(value, Unknown)


@plugin
def validate_type(
    fq_type_name: "string", value: "any", validation_parameters: "dict" = None
) -> "bool":
    """
        Check whether `value` satisfies the constraints of type `fq_type_name`. When the given type (fq_type_name)
        requires validation_parameters, they can be provided using the optional `validation_parameters` argument.

        The following types require validation_parameters:

            * pydantic.condecimal:
                gt: Decimal = None
                ge: Decimal = None
                lt: Decimal = None
                le: Decimal = None
                max_digits: int = None
                decimal_places: int = None
                multiple_of: Decimal = None
            * pydantic.confloat and pydantic.conint:
                gt: float = None
                ge: float = None
                lt: float = None
                le: float = None
                multiple_of: float = None,
            * pydantic.constr:
                min_length: int = None
                max_length: int = None
                curtail_length: int = None (Only verify the regex on the first curtail_length characters)
                regex: str = None          (The regex is verified via Pattern.match())
            * pydantic.stricturl:
                min_length: int = 1
                max_length: int = 2 ** 16
                tld_required: bool = True
                allowed_schemes: Optional[Set[str]] = None

        Example usage:

            * Define a vlan_id type which represent a valid vlan ID (0-4,095):

              typedef vlan_id as number matching std::validate_type("pydantic.conint", self, {"ge": 0, "le": 4095})


    """
    if not (
        fq_type_name.startswith("pydantic.")
        or fq_type_name.startswith("datetime.")
        or fq_type_name.startswith("ipaddress.")
        or fq_type_name.startswith("uuid.")
    ):
        return False
    module_name, type_name = fq_type_name.split(".", 1)
    module = importlib.import_module(module_name)
    t = getattr(module, type_name)
    # Construct pydantic model
    if validation_parameters is not None:
        model = pydantic.create_model(
            fq_type_name, value=(t(**validation_parameters), ...)
        )
    else:
        model = pydantic.create_model(fq_type_name, value=(t, ...))
    # Do validation
    try:
        model(value=value)
    except pydantic.ValidationError:
        return False

    return True


@plugin
def is_base64_encoded(s: "string") -> "bool":
    """
        Check whether the given string is base64 encoded.
    """
    try:
        encoded_str = s.encode("utf-8")
        base64.b64decode(encoded_str, validate=True)
    except Exception:
        return False
    return True
