"""测试主循环的解析逻辑与异常处理。"""

import pytest

from agent_runtime import AgentRuntime, MAX_CONSECUTIVE_TOOL_ERRORS


class TestParseOutput:
    def test_plain_json(self):
        raw = '{"thought": "t", "action": "final_answer", "final_answer": "ok"}'
        parsed = AgentRuntime._parse_output(raw)
        assert parsed["final_answer"] == "ok"

    def test_json_with_code_block(self):
        raw = '```json\n{"thought": "t", "action": "final_answer", "final_answer": "ok"}\n```'
        parsed = AgentRuntime._parse_output(raw)
        assert parsed["action"] == "final_answer"

    def test_extract_json_from_extra_text(self):
        raw = '前面有文字 {"thought": "t", "action": "final_answer", "final_answer": "ok"} 后面有文字'
        parsed = AgentRuntime._parse_output(raw)
        assert parsed["final_answer"] == "ok"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            AgentRuntime._parse_output("这不是 JSON")


class TestMainLoop:
    def test_final_answer_directly(self, registry, make_llm):
        llm = make_llm(
            ['{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "你好"}']
        )
        agent = AgentRuntime(llm, registry)
        assert agent.run("打招呼") == "你好"
        assert len(llm.calls) == 1

    def test_tool_then_final_answer(self, registry, make_llm):
        llm = make_llm(
            [
                '{"thought": "需要计算", "action": "calculator", "action_input": {"expression": "3+4"}, "final_answer": null}',
                '{"thought": "得到结果", "action": "final_answer", "action_input": null, "final_answer": "结果是7"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        assert agent.run("3+4") == "结果是7"
        # 第二次调用应包含工具结果
        second_messages = llm.calls[1]["messages"]
        joined = " ".join(m["content"] for m in second_messages)
        assert "7" in joined


class TestToolErrorHandling:
    def test_tool_error_prefix(self, registry, make_llm):
        # calculator 非法表达式会抛 ValueError
        llm = make_llm(
            [
                '{"thought": "t", "action": "calculator", "action_input": {"expression": "1/0"}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "请重新输入"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        result = agent.run("计算", session_id="s1")
        assert result == "请重新输入"
        # session 中应有 [Tool Error] 前缀
        hist = agent.session_manager.get_session("s1").get_history()
        contents = [m["content"] for m in hist]
        assert any(c.startswith("Observation: [Tool Error]") for c in contents)

    def test_tool_error_not_propagated(self, registry, make_llm):
        llm = make_llm(
            [
                '{"thought": "t", "action": "todo_delete", "action_input": {"id": 999}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "ok"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        # 不抛异常即通过
        agent.run("删除", session_id="s1")

    def test_consecutive_errors_inject_apology(self, registry, make_llm):
        llm = make_llm(
            [
                '{"thought": "t", "action": "todo_delete", "action_input": {"id": 999}, "final_answer": null}',
                '{"thought": "t", "action": "todo_delete", "action_input": {"id": 999}, "final_answer": null}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "抱歉，请换个方式"}',
            ]
        )
        agent = AgentRuntime(llm, registry, max_iterations=5)
        result = agent.run("删除", session_id="s1")
        assert result == "抱歉，请换个方式"
        third = llm.calls[2]["messages"]
        assert any("工具连续报错" in m["content"] for m in third)


class TestFatalErrorHandling:
    def test_llm_failure_fallback(self, registry, make_llm):
        llm = make_llm([Exception("connection timeout")])
        agent = AgentRuntime(llm, registry)
        result = agent.run("你好", session_id="s1")
        assert "暂时无法连接服务器" in result

    def test_llm_failure_writes_system_error(self, registry, make_llm):
        llm = make_llm([Exception("connection timeout")])
        agent = AgentRuntime(llm, registry)
        agent.run("你好", session_id="s1")
        hist = agent.session_manager.get_session("s1").get_history()
        roles = [m["role"] for m in hist]
        assert "system" in roles
        contents = [m["content"] for m in hist]
        assert not any("connection timeout" in c for c in contents)

    def test_parse_error_fallback(self, registry, make_llm):
        llm = make_llm(["完全无法解析的乱码###"])
        agent = AgentRuntime(llm, registry)
        result = agent.run("你好", session_id="s1")
        assert "换个方式描述" in result
