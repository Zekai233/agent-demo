"""测试连续对话中 context 压缩是否生效。"""

import pytest

from agent_runtime import AgentRuntime


class TestContextCompression:
    def test_compress_triggers_when_long(self, registry, make_llm):
        """超长历史应触发压缩，用摘要替代旧历史并保留最近消息。"""
        long_history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
            {"role": "user", "content": "z" * 1000},
            {"role": "assistant", "content": "w" * 1000},
        ]
        # 第一次 chat 用于摘要，第二次用于主循环
        llm = make_llm(
            [
                "这是历史摘要",
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "完成"}',
            ]
        )
        agent = AgentRuntime(
            llm, registry, max_context_chars=100, keep_recent_messages=2
        )
        compressed = agent._maybe_compress(long_history)

        assert len(compressed) < len(long_history)
        assert compressed[0]["role"] == "system"
        assert "摘要" in compressed[0]["content"]
        assert compressed[-2:] == long_history[-2:]

    def test_compress_not_triggered_when_short(self, registry, make_llm):
        short_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        llm = make_llm([])
        agent = AgentRuntime(llm, registry, max_context_chars=1000)
        result = agent._maybe_compress(short_history)
        assert result == short_history  # 不触发压缩，原样返回

    def test_compress_keeps_tool_results(self, registry, make_llm):
        """压缩后应保留最近的工具执行结果。"""
        history = [
            {"role": "user", "content": "旧问题" * 500},
            {"role": "assistant", "content": "旧回答" * 500},
            {"role": "user", "content": "Observation: 工具 'calculator' 返回: 42"},
            {"role": "assistant", "content": '{"action": "final_answer", "final_answer": "..."}'},
        ]
        llm = make_llm(["摘要文本"])
        agent = AgentRuntime(llm, registry, max_context_chars=100, keep_recent_messages=2)

        compressed = agent._maybe_compress(history)
        contents = [m["content"] for m in compressed]
        assert any("工具 'calculator' 返回: 42" in c for c in contents)

    def test_compress_summary_failure_falls_back(self, registry, make_llm):
        long_history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
            {"role": "user", "content": "z" * 1000},
            {"role": "assistant", "content": "w" * 1000},
        ]
        llm = make_llm([Exception("摘要服务不可用")])
        agent = AgentRuntime(llm, registry, max_context_chars=100, keep_recent_messages=2)

        compressed = agent._maybe_compress(long_history)
        # 退化为截断，不抛异常
        assert compressed[-2:] == long_history[-2:]


class TestContinuousConversation:
    def test_tool_result_in_followup(self, registry, make_llm):
        """带工具的连续对话，之前工具结果应正确进入后续上下文。"""
        llm = make_llm(
            [
                '{"thought": "t", "action": "calculator", "action_input": {"expression": "3+4"}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "结果是7"}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "刚才算的是7"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        agent.run("3+4", session_id="s1")
        agent.run("刚才结果是多少", session_id="s1")

        third_messages = llm.calls[2]["messages"]
        contents = [m["content"] for m in third_messages]
        assert any("工具 'calculator' 返回: 7" in c for c in contents)

    def test_compress_triggered_in_long_conversation(self, registry, make_llm):
        """模拟长对话，验证压缩在真实循环中被触发。"""
        # 第 1 轮：工具调用；第 2 轮前触发压缩（消耗 1 个摘要响应），再产出最终答复
        llm = make_llm(
            [
                '{"thought": "t", "action": "calculator", "action_input": {"expression": "1+1"}, "final_answer": null}',
                "历史摘要",  # _summarize 消耗
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "完成"}',
            ]
        )
        agent = AgentRuntime(
            llm, registry, max_context_chars=10, keep_recent_messages=2
        )
        result = agent.run("计算", session_id="s1")
        assert result == "完成"
        # 验证确实发生了压缩（有一条摘要 system 消息进入了某次调用）
        any_summary = False
        for call in llm.calls:
            for m in call["messages"]:
                if m["role"] == "system" and "摘要" in m["content"]:
                    any_summary = True
        assert any_summary
