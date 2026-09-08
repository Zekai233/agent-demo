"""测试工具注册机制与三个具象工具的基本功能。"""

import pytest

from tool_registry import ToolRegistry, tool
from tools import register_all_tools


class TestToolRegistry:
    def test_register_and_schema(self):
        reg = ToolRegistry()

        @tool(
            reg,
            name="add",
            description="计算两个数的和",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        def add(a, b):
            return a + b

        schemas = reg.get_all_tools_schema()
        assert len(schemas) == 1
        fn = schemas[0]["function"]
        assert fn["name"] == "add"
        assert fn["description"] == "计算两个数的和"
        assert fn["parameters"]["type"] == "object"

    def test_duplicate_register_raises(self):
        reg = ToolRegistry()

        @tool(reg, description="x", parameters={"type": "object"})
        def foo():
            return 1

        with pytest.raises(ValueError):
            tool(reg, description="x", parameters={"type": "object"})(foo)

    def test_execute_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.execute("nonexistent", {})

    def test_register_all_tools(self, registry):
        names = {s["function"]["name"] for s in registry.get_all_tools_schema()}
        assert names == {"calculator", "search", "todo_add", "todo_list", "todo_delete"}


class TestCalculator:
    def test_basic(self, registry):
        assert registry.execute("calculator", {"expression": "3 + 4 * 2"}) == 11

    def test_math_function(self, registry):
        assert registry.execute("calculator", {"expression": "sqrt(16)"}) == 4

    def test_power(self, registry):
        assert registry.execute("calculator", {"expression": "2 ** 10"}) == 1024

    def test_invalid_expression(self, registry):
        with pytest.raises(Exception):
            registry.execute("calculator", {"expression": "1 / 0"})

    def test_no_dangerous_eval(self, registry):
        with pytest.raises(Exception):
            registry.execute(
                "calculator", {"expression": "__import__('os').system('ls')"}
            )


class TestSearch:
    def test_hit(self, registry):
        result = registry.execute("search", {"query": "python"})
        assert any("Python" in r["title"] for r in result)

    def test_miss(self, registry):
        assert registry.execute("search", {"query": "不存在的词xyz"}) == []

    def test_case_insensitive(self, registry):
        result = registry.execute("search", {"query": "PYTHON"})
        assert len(result) > 0


class TestTodo:
    def test_add(self, registry, reset_todo):
        item = registry.execute("todo_add", {"content": "写周报"})
        assert item["content"] == "写周报"
        assert item["id"] == 1

    def test_list(self, registry, reset_todo):
        registry.execute("todo_add", {"content": "任务A"})
        registry.execute("todo_add", {"content": "任务B"})
        assert len(registry.execute("todo_list", {})) == 2

    def test_delete(self, registry, reset_todo):
        registry.execute("todo_add", {"content": "任务A"})
        result = registry.execute("todo_delete", {"id": 1})
        assert result["deleted"]["content"] == "任务A"
        assert registry.execute("todo_list", {}) == []

    def test_delete_not_found(self, registry, reset_todo):
        with pytest.raises(ValueError):
            registry.execute("todo_delete", {"id": 999})
