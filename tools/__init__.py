"""具象工具包：统一提供工具注册入口。

用法:
    from tool_registry import ToolRegistry
    from tools import register_all_tools

    registry = ToolRegistry()
    register_all_tools(registry)
"""

from tool_registry import ToolRegistry

from . import calculator, search, todo


def register_all_tools(registry: ToolRegistry) -> None:
    """将全部具象工具注册到给定注册表。"""
    calculator.register(registry)
    search.register(registry)
    todo.register(registry)


__all__ = ["register_all_tools", "calculator", "search", "todo"]
