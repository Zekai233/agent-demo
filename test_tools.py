"""tools.py 工具注册机制的单元测试。"""

import unittest

from tools import ToolRegistry, tool


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def _register_calc(self):
        @tool(
            self.registry,
            description="计算两个数的和",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
        )
        def add(a, b):
            return a + b

        return add

    def test_register_and_execute(self):
        self._register_calc()
        self.assertEqual(self.registry.execute("add", {"a": 1, "b": 2}), 3)

    def test_execute_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            self.registry.execute("not_exist", {})

    def test_duplicate_register_raises(self):
        self._register_calc()
        with self.assertRaises(ValueError):
            self._register_calc()

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register("", "desc", {"type": "object"}, lambda: None)

    def test_get_all_tools_schema(self):
        self._register_calc()
        schemas = self.registry.get_all_tools_schema()

        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["type"], "function")
        fn = schemas[0]["function"]
        self.assertEqual(fn["name"], "add")
        self.assertEqual(fn["description"], "计算两个数的和")
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertIn("a", fn["parameters"]["properties"])

    def test_default_name_uses_func_name(self):
        @tool(self.registry)
        def hello():
            """打招呼"""
            return "hi"

        schemas = self.registry.get_all_tools_schema()
        self.assertEqual(schemas[0]["function"]["name"], "hello")
        # description 缺省时取 docstring 首行
        self.assertEqual(schemas[0]["function"]["description"], "打招呼")

    def test_default_parameters(self):
        @tool(self.registry)
        def noop():
            pass

        schemas = self.registry.get_all_tools_schema()
        self.assertEqual(
            schemas[0]["function"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_multiple_tools(self):
        self._register_calc()

        @tool(
            self.registry,
            name="greet",
            description="打招呼",
            parameters={"type": "object", "properties": {}},
        )
        def greet(name):
            return f"Hello, {name}"

        schemas = self.registry.get_all_tools_schema()
        self.assertEqual(len(schemas), 2)

        names = {s["function"]["name"] for s in schemas}
        self.assertEqual(names, {"add", "greet"})
        self.assertEqual(self.registry.execute("greet", {"name": "Tom"}), "Hello, Tom")


if __name__ == "__main__":
    unittest.main()
