"""pytest 全局 fixtures。

提供统一的 ToolRegistry、注册好的工具集、以及可模拟的 LLM。
"""

import pytest

from tool_registry import ToolRegistry
from tools import register_all_tools
from tools import todo as todo_mod


@pytest.fixture
def registry():
    """返回一个已注册全部具象工具的注册表。"""
    reg = ToolRegistry()
    register_all_tools(reg)
    return reg


@pytest.fixture
def reset_todo():
    """重置 todo 内存存储，保证用例隔离。"""
    todo_mod._reset()
    yield
    todo_mod._reset()


class FakeLLM:
    """可编程的 LLM 替身：按预设顺序返回响应，记录每次调用参数。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, system_prompt=""):
        self.calls.append(
            {"messages": list(messages), "system_prompt": system_prompt}
        )
        if not self.responses:
            raise RuntimeError("FakeLLM 响应已耗尽")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def make_llm():
    """工厂：生成 FakeLLM 实例。"""
    return lambda responses: FakeLLM(responses)
