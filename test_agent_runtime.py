"""agent_runtime.py 的单元测试（mock LLM，不依赖真实 API）。"""

import unittest
from unittest.mock import MagicMock

from agent_runtime import AgentRuntime, SYSTEM_PROMPT_TEMPLATE
from tool_registry import ToolRegistry, tool


def _make_llm(responses):
    """构造一个按顺序返回固定响应的 mock LLM。"""
    llm = MagicMock()
    llm.chat.side_effect = list(responses)
    return llm


class TestAgentRuntime(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

        @tool(
            self.registry,
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

    def test_final_answer_directly(self):
        llm = _make_llm(
            [
                '{"thought": "不需要工具", "action": "final_answer", '
                '"action_input": null, "final_answer": "你好"}'
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("打个招呼")
        self.assertEqual(result, "你好")
        # 只调用了一次 LLM
        self.assertEqual(llm.chat.call_count, 1)

    def test_tool_then_final_answer(self):
        llm = _make_llm(
            [
                '{"thought": "需要计算", "action": "add", '
                '"action_input": {"a": 1, "b": 2}, "final_answer": null}',
                '{"thought": "得到结果", "action": "final_answer", '
                '"action_input": null, "final_answer": "结果是3"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("1+2 等于几")
        self.assertEqual(result, "结果是3")
        self.assertEqual(llm.chat.call_count, 2)

        # 验证第二次调用时，上下文里包含 observation
        second_call_messages = llm.chat.call_args_list[1].kwargs["messages"]
        joined = " ".join(m["content"] for m in second_call_messages)
        self.assertIn("Observation", joined)
        self.assertIn("3", joined)

    def test_parse_output_with_code_block(self):
        raw = '```json\n{"thought": "t", "action": "final_answer", "final_answer": "ok"}\n```'
        parsed = AgentRuntime._parse_output(raw)
        self.assertEqual(parsed["final_answer"], "ok")

    def test_tool_error_fed_back_to_llm(self):
        # 第一次调用一个不存在的工具，会抛 KeyError，需反馈给 LLM
        llm = _make_llm(
            [
                '{"thought": "调工具", "action": "not_exist", '
                '"action_input": {}, "final_answer": null}',
                '{"thought": "工具报错", "action": "final_answer", '
                '"action_input": null, "final_answer": "fallback"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("测试")
        self.assertEqual(result, "fallback")

        # 第二次 LLM 调用上下文中应包含错误信息
        second_call_messages = llm.chat.call_args_list[1].kwargs["messages"]
        joined = " ".join(m["content"] for m in second_call_messages)
        self.assertIn("执行出错", joined)

    def test_max_iterations_stop(self):
        # LLM 永远要求调用工具，应被 max_iterations 截断
        llm = _make_llm(
            ['{"thought": "x", "action": "add", "action_input": {"a":1,"b":1}, "final_answer": null}']
            * 10
        )
        agent = AgentRuntime(llm, self.registry, max_iterations=3)
        result = agent.run("循环")
        self.assertIn("最大循环次数", result)
        self.assertEqual(llm.chat.call_count, 3)

    def test_empty_action_prompted(self):
        llm = _make_llm(
            [
                '{"thought": "x", "action": "", "action_input": null, "final_answer": null}',
                '{"thought": "x", "action": "final_answer", "action_input": null, "final_answer": "done"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry, max_iterations=5)
        result = agent.run("测试")
        self.assertEqual(result, "done")


if __name__ == "__main__":
    unittest.main()
