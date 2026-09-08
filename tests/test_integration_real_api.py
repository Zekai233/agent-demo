"""真实 LLM API 集成测试。

这些测试会真实调用 DeepSeek API（消耗 token），默认不被 `pytest tests/` 收集，
需要显式指定运行：

    pytest tests/test_integration_real_api.py -m integration -v

前置条件：
- .env 中已配置 DEEPSEEK_API_KEY
- 网络可访问 api.deepseek.com
"""

import os

import pytest

from agent_runtime import AgentRuntime
from llm_client import LLMClient
from tool_registry import ToolRegistry
from tools import register_all_tools
from tools import todo as todo_mod

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def agent():
    """创建一个已连接真实 LLM 的 Agent。"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("未配置 DEEPSEEK_API_KEY，跳过真实 API 测试")
    reg = ToolRegistry()
    register_all_tools(reg)
    llm = LLMClient()
    return AgentRuntime(llm, reg, max_iterations=6)


class TestRealCalculator:
    def test_calculator(self, agent):
        """真实 API：通过 calculator 工具计算乘法。"""
        answer = agent.run("请帮我计算 123 乘以 456 等于多少", session_id="real-calc")
        assert "56088" in answer


class TestRealSearch:
    def test_search(self, agent):
        """真实 API：通过 search 工具搜索 python 资料。"""
        answer = agent.run("请帮我搜索一下 python 相关的资料", session_id="real-search")
        assert "python" in answer.lower()


class TestRealTodo:
    def test_todo_add(self, agent):
        """真实 API：通过 todo_add 工具添加待办，并验证真实写入存储。"""
        todo_mod._reset()
        answer = agent.run("帮我添加一条待办事项：明天下午开会", session_id="real-todo")
        assert "明天下午开会" in answer
        # 验证工具真实执行，存储中确实新增了记录
        items = todo_mod._get_all()
        assert any("明天下午开会" in i["content"] for i in items)
