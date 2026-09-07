"""agent_runtime.py 的单元测试（mock LLM，不依赖真实 API）。"""

import unittest
from unittest.mock import MagicMock

from agent_runtime import (
    AgentRuntime,
    SYSTEM_PROMPT_TEMPLATE,
    FatalError,
    MAX_CONSECUTIVE_TOOL_ERRORS,
)
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

    def test_session_history_persisted(self):
        """传入 session_id 时，对话历史应写回对应 session。"""
        llm = _make_llm(
            [
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "hello"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        agent.run("第一问", session_id="u1")

        session = agent.session_manager.get_session("u1")
        hist = session.get_history()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0], {"role": "user", "content": "第一问"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "hello"})

    def test_session_context_loaded_on_next_turn(self):
        """第二次调用同一 session，应将历史作为上下文传给 LLM。"""
        llm = _make_llm(
            [
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "第一次回答"}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "第二次回答"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        agent.run("问题1", session_id="u1")
        agent.run("问题2", session_id="u1")

        # 第二次调用 LLM 时，messages 应包含第一次的历史
        second_call_messages = llm.chat.call_args_list[1].kwargs["messages"]
        contents = [m["content"] for m in second_call_messages]
        self.assertIn("问题1", contents)  # 历史 user 消息
        self.assertIn("问题2", contents)  # 本轮 user 消息

    def test_session_isolation(self):
        """不同 session 的历史应完全隔离。"""
        llm = _make_llm(
            [
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "A答"}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "B答"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        agent.run("A的问题", session_id="A")
        agent.run("B的问题", session_id="B")

        a_hist = agent.session_manager.get_session("A").get_history()
        b_hist = agent.session_manager.get_session("B").get_history()
        self.assertEqual(a_hist[0]["content"], "A的问题")
        self.assertEqual(b_hist[0]["content"], "B的问题")
        self.assertNotEqual(a_hist[0]["content"], b_hist[0]["content"])


class TestErrorHandling(unittest.TestCase):
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
            # 当 b 为负数时抛可恢复错误，模拟工具执行异常
            if b < 0:
                raise ValueError("b 不能为负数")
            return a + b

    def test_tool_error_prefix_written_to_session(self):
        """工具异常应转成 [Tool Error] 前缀并写回 session。"""
        llm = _make_llm(
            [
                '{"thought": "t", "action": "add", "action_input": {"a": 1, "b": -1}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "抱歉，请重新输入"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("计算", session_id="s1")
        self.assertEqual(result, "抱歉，请重新输入")

        # session 中应包含 [Tool Error] 前缀的消息
        hist = agent.session_manager.get_session("s1").get_history()
        contents = [m["content"] for m in hist]
        self.assertTrue(any(c.startswith("Observation: [Tool Error]") for c in contents))

    def test_tool_error_not_propagated(self):
        """工具异常不应抛出到 run() 之外，而是被捕获。"""
        llm = _make_llm(
            [
                '{"thought": "t", "action": "add", "action_input": {"a": 1, "b": -1}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "ok"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry)
        # 不抛异常即为通过
        agent.run("计算", session_id="s1")

    def test_consecutive_error_injects_apology(self):
        """同一工具连续报错超阈值，应注入致歉提示。"""
        # 连续两次都调用 add 且都报错
        llm = _make_llm(
            [
                '{"thought": "t", "action": "add", "action_input": {"a": 1, "b": -1}, "final_answer": null}',
                '{"thought": "t", "action": "add", "action_input": {"a": 1, "b": -2}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "抱歉，请换个方式提问"}',
            ]
        )
        agent = AgentRuntime(llm, self.registry, max_iterations=5)
        result = agent.run("计算", session_id="s1")
        self.assertEqual(result, "抱歉，请换个方式提问")

        # 第三次 LLM 调用的上下文中应包含致歉提示
        third_call = llm.chat.call_args_list[2].kwargs["messages"]
        contents = [m["content"] for m in third_call]
        self.assertTrue(any("工具连续报错" in c for c in contents))

    def test_fatal_llm_error_returns_fallback(self):
        """LLM 调用失败（致命错误）应返回降级提示并终止。"""
        llm = MagicMock()
        llm.chat.side_effect = Exception("connection timeout")
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("你好", session_id="s1")
        self.assertIn("暂时无法连接服务器", result)

    def test_fatal_llm_error_writes_system_error_to_session(self):
        """致命错误应在 session 中记录 System Error 消息，而非原始错误。"""
        llm = MagicMock()
        llm.chat.side_effect = Exception("connection timeout")
        agent = AgentRuntime(llm, self.registry)
        agent.run("你好", session_id="s1")

        hist = agent.session_manager.get_session("s1").get_history()
        roles = [m["role"] for m in hist]
        self.assertIn("system", roles)
        # 不应包含原始错误细节
        contents = [m["content"] for m in hist]
        self.assertFalse(any("connection timeout" in c for c in contents))

    def test_fatal_parse_error_returns_fallback(self):
        """LLM 输出无法解析（致命错误）应返回降级提示。"""
        llm = _make_llm(["这不是一个合法的JSON输出，全是乱码###"])
        agent = AgentRuntime(llm, self.registry)
        result = agent.run("你好", session_id="s1")
        self.assertIn("换个方式描述", result)


if __name__ == "__main__":
    unittest.main()
