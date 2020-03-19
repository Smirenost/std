"""
    Copyright 2017 Inmanta

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
import os
import shutil

import pytest


def test_is_defined(project):
    """
        Test the use of is defined
    """
    project.add_mock_file(
        "templates",
        "testtemplate.tmpl",
        "{% if other is defined %} {{name}} : {{ other.name }} "
        "{% if other.other is defined %} sub: {{ other.other.name }} {% endif %} "
        "{% else %} {{name}} is not defined {% endif %}"
        "",
    )

    project.compile(
        """
import std
import unittest

entity Test1:
string name
end

Test1 prev [0:1] -- [0:1] Test1 other

implementation tt for Test1:
content=std::template("unittest/testtemplate.tmpl")
std::print(content)
end

implement Test1 using tt when not(self.prev is defined)
implement Test1 using std::none when self.prev is defined

Test1(name="t1",other=Test1(name="t11"))
Test1(name="t2")
Test1(name="t3",other=Test1(name="t31",other=Test1(name="t32")))
    """
    )

    assert "t3 : t31  sub: t32" in project.get_stdout()
    assert "t1 : t11" in project.get_stdout()
    assert "t2 is not defined" in project.get_stdout()


def test_template(project):
    """
        Test the evaluation of a template
    """
    project.add_mock_file("templates", "test.tmpl", "{{ value }}")
    project.compile(
        """import unittest
value = "1234"
std::print(std::template("unittest/test.tmpl"))
    """
    )

    assert project.get_stdout() == "1234\n"


def test_plugin_with_list(project):
    """
        Test the use of is defined
    """
    project.add_mock_file(
        "templates",
        "test.tmpl",
        """{% for item in items | std.key_sort("name") %}  {{ item.name }}
{% endfor %}""",
    )

    project.compile(
        """
import std
import unittest

entity Item:
    string name
end

implement Item using std::none

entity Collection:
    string content
end

implementation makeContent for Collection:
    self.content = std::template("unittest/test.tmpl")
end

implement Collection using makeContent

Collection.items [0:] -- Item.collection [0:]

c1 = Collection()

t1 = Item(name="t1", collection=c1)
t2 = Item(name="t2", collection=c1)
t3 = Item(name="t3", collection=c1)
    """
    )


@pytest.fixture()
def module_with_template(project):
    module_init_cf = """
    import std

    entity Test1:
    string name
    end

    implementation tt for Test1:
    content=std::template("./testtemplate.tmpl")
    std::print(content)
    end

    implement Test1 using tt

    Test1(name="t1")
    """
    project.create_module("testmod", initcf=module_init_cf, initpy="")
    template_dir = os.path.join(
        project._test_project_dir, "libs", "testmod", "templates"
    )
    with open(os.path.join(template_dir, "testtemplate.tmpl"), "w") as template_file:
        template_file.write("""{{name}}""")
    yield
    shutil.rmtree(os.path.join(project._test_project_dir, "libs", "testmod"))


def test_template_current_dir(project, module_with_template):

    """
        Test the use of current dir in templates
    """

    project.compile(
        """
import testmod
        """
    )

    assert project.get_stdout() == "t1\n"


@pytest.fixture()
def module_with_file(project):
    module_init_cf = """
        import std

        for file in std::list_files("./"):
                std::source("./{{file}}")
                std::print(file)
            end
        """

    project.create_module("testmod", initcf=module_init_cf, initpy="")
    files_dir = os.path.join(project._test_project_dir, "libs", "testmod", "files")
    with open(os.path.join(files_dir, "testfile1"), "w") as template_file:
        template_file.write("test test test")
    yield
    shutil.rmtree(os.path.join(project._test_project_dir, "libs", "testmod"))


def test_files_current_dir(project, module_with_file):
    project.compile(
        """
        import testmod
        """
    )

    out = project.get_stdout().splitlines()
    assert ["testfile1"] == out
