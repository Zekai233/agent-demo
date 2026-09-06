"""工具注册机制。

提供 ToolRegistry 类与 @tool 装饰器，用于注册、描述与执行工具。
每个工具包含 name、description、parameters（JSON Schema 格式）。
"""

import inspect
from typing import Any, Callable, Dict, List


class ToolRegistry:
    """工具注册表：管理工具的定义、Schema 导出与执行。"""

    def __init__(self) -> None:
        # name -> 工具元信息（含可调用函数）
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ) -> None:
        """注册一个工具。

        参数:
            name: 工具唯一名称。
            description: 工具功能描述（会发给 LLM）。
            parameters: JSON Schema 格式的参数定义。
            func: 实际执行的函数。
        """
        if not name:
            raise ValueError("工具名称不能为空")
        if name in self._tools:
            raise ValueError(f"工具 '{name}' 已存在，请勿重复注册")
        if not isinstance(parameters, dict):
            raise ValueError("parameters 必须是 dict（JSON Schema 格式）")

        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func,
        }

    def get_all_tools_schema(self) -> List[Dict[str, Any]]:
        """返回所有工具的 OpenAI function-calling 格式定义列表。

        返回:
            list[dict]: 每项形如
                {"type": "function", "function": {name, description, parameters}}
            可直接作为 LLM 请求中的 tools 参数。
        """
        schemas: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return schemas

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """查找并执行指定工具。

        参数:
            tool_name: 工具名称。
            arguments: 传给工具的参数字典。

        返回:
            Any: 工具函数的返回值。

        异常:
            KeyError: 工具不存在时抛出。
        """
        if tool_name not in self._tools:
            raise KeyError(f"工具 '{tool_name}' 未注册")
        return self._tools[tool_name]["func"](**arguments)


def tool(
    registry: ToolRegistry,
    name: str = "",
    description: str = "",
    parameters: Dict[str, Any] = None,
) -> Callable:
    """工具注册装饰器。

    用法:
        @tool(registry, description="...", parameters={...})
        def my_tool(a: int, b: int) -> int:
            ...

    参数:
        registry: 目标 ToolRegistry 实例。
        name: 工具名，缺省时使用函数名。
        description: 工具描述，缺省时使用函数 docstring 首行。
        parameters: JSON Schema 参数定义，缺省时使用空 schema。
    """
    if parameters is None:
        parameters = {"type": "object", "properties": {}}

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "").strip().split("\n")[0]
        registry.register(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            func=func,
        )
        return func

    return decorator


# 全局默认注册表，便于直接使用
default_registry = ToolRegistry()
