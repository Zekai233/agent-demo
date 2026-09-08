"""测试 Session 的隔离性。"""

import pytest

from agent_runtime import AgentRuntime
from session import Session, SessionManager


class TestSession:
    def test_add_and_get(self):
        s = Session("sid")
        s.add_message("user", "你好")
        s.add_message("assistant", "你好呀")
        assert len(s.get_history()) == 2

    def test_get_history_returns_copy(self):
        s = Session("sid")
        s.add_message("user", "hi")
        hist = s.get_history()
        hist.append({"role": "user", "content": "外部修改"})
        assert len(s.messages) == 1  # 内部状态不受影响


class TestSessionManager:
    def test_create_get_destroy(self):
        m = SessionManager()
        m.create_session(session_id="abc")
        assert m.get_session("abc").session_id == "abc"
        assert m.destroy_session("abc") is True
        assert m.destroy_session("abc") is False

    def test_get_or_create(self):
        m = SessionManager()
        s1 = m.get_or_create("u1")
        s2 = m.get_or_create("u1")
        assert s1 is s2

    def test_isolation_between_sessions(self):
        m = SessionManager()
        a = m.create_session(session_id="A")
        b = m.create_session(session_id="B")
        a.add_message("user", "A 的问题")
        b.add_message("user", "B 的问题")
        assert a.messages[0]["content"] == "A 的问题"
        assert b.messages[0]["content"] == "B 的问题"
        assert a.messages[0]["content"] != b.messages[0]["content"]


class TestSessionIsolationInRuntime:
    def test_runtime_multi_session_isolated(self, registry, make_llm):
        """两个 session 的对话历史应互不影响。"""
        llm = make_llm(
            [
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "A答"}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "B答"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        agent.run("A的问题", session_id="A")
        agent.run("B的问题", session_id="B")

        a_hist = agent.session_manager.get_session("A").get_history()
        b_hist = agent.session_manager.get_session("B").get_history()
        assert a_hist[0]["content"] == "A的问题"
        assert b_hist[0]["content"] == "B的问题"

    def test_runtime_second_call_loads_history(self, registry, make_llm):
        """同一 session 第二次调用应带上第一次的历史。"""
        llm = make_llm(
            [
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "第一答"}',
                '{"thought": "t", "action": "final_answer", "action_input": null, "final_answer": "第二答"}',
            ]
        )
        agent = AgentRuntime(llm, registry)
        agent.run("问题1", session_id="u1")
        agent.run("问题2", session_id="u1")

        second_messages = llm.calls[1]["messages"]
        contents = [m["content"] for m in second_messages]
        assert "问题1" in contents
        assert "问题2" in contents
