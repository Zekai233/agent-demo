"""calculator 工具：安全计算数学表达式。

使用 asteval 库安全求值，禁止使用 Python 内置 eval。
"""

from typing import Any

from asteval import Interpreter

from tool_registry import ToolRegistry, tool

# 安全的表达式求值器：仅支持数学运算与白名单函数，无系统/文件访问能力
_evaluator = Interpreter(
    minimal=True,  # 最小符号表，移除危险内置
    use_numpy=False,
    max_statement_length=200,
)

CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "要计算的数学表达式，例如 '3 + 4 * 2' 或 'sqrt(16) + 2**3'",
        }
    },
    "required": ["expression"],
}


def register(registry: ToolRegistry) -> None:
    """将 calculator 工具注册到给定注册表。"""

    @tool(
        registry,
        name="calculator",
        description="安全计算数学表达式，支持 + - * / ** 及 sqrt、sin、cos、log 等数学函数",
        parameters=CALCULATOR_SCHEMA,
    )
    def calculator(expression: str) -> Any:
        """计算数学表达式，返回数值结果。"""
        result = _evaluator(expression)
        if _evaluator.error:
            # 表达式非法时返回错误信息，供 LLM 理解并纠正
            raise ValueError(f"表达式解析失败: {_evaluator.error[0].get_error()[1]}")
        return result
