"""tools/ 目录三个具象工具的单元测试。"""

import unittest

from tool_registry import ToolRegistry
from tools import register_all_tools
from tools import todo as todo_mod


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_all_tools(self.registry)

    def test_basic_arithmetic(self):
        self.assertEqual(self.registry.execute("calculator", {"expression": "3 + 4 * 2"}), 11)

    def test_math_function(self):
        self.assertEqual(self.registry.execute("calculator", {"expression": "sqrt(16)"}), 4)

    def test_power(self):
        self.assertEqual(self.registry.execute("calculator", {"expression": "2 ** 10"}), 1024)

    def test_invalid_expression_raises(self):
        with self.assertRaises(Exception):
            self.registry.execute("calculator", {"expression": "1 / 0"})

    def test_no_dangerous_eval(self):
        # 尝试访问系统，应被安全求值器拒绝
        with self.assertRaises(Exception):
            self.registry.execute("calculator", {"expression": "__import__('os').system('ls')"})


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_all_tools(self.registry)

    def test_search_hit(self):
        result = self.registry.execute("search", {"query": "python"})
        self.assertTrue(any("Python" in r["title"] for r in result))

    def test_search_miss(self):
        result = self.registry.execute("search", {"query": "不存在的关键词xyz"})
        self.assertEqual(result, [])


class TestTodo(unittest.TestCase):
    def setUp(self):
        todo_mod._reset()  # 每个用例前重置存储
        self.registry = ToolRegistry()
        register_all_tools(self.registry)

    def test_add(self):
        item = self.registry.execute("todo_add", {"content": "写周报"})
        self.assertEqual(item["content"], "写周报")
        self.assertEqual(item["id"], 1)

    def test_list(self):
        self.registry.execute("todo_add", {"content": "任务A"})
        self.registry.execute("todo_add", {"content": "任务B"})
        items = self.registry.execute("todo_list", {})
        self.assertEqual(len(items), 2)

    def test_delete(self):
        self.registry.execute("todo_add", {"content": "任务A"})
        result = self.registry.execute("todo_delete", {"id": 1})
        self.assertEqual(result["deleted"]["content"], "任务A")
        self.assertEqual(self.registry.execute("todo_list", {}), [])

    def test_delete_not_found(self):
        with self.assertRaises(ValueError):
            self.registry.execute("todo_delete", {"id": 999})


class TestRegisterAll(unittest.TestCase):
    def test_all_tools_registered(self):
        registry = ToolRegistry()
        register_all_tools(registry)
        names = {s["function"]["name"] for s in registry.get_all_tools_schema()}
        expected = {"calculator", "search", "todo_add", "todo_list", "todo_delete"}
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()
