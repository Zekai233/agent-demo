"""session.py 会话管理的单元测试。"""

import unittest

from session import Session, SessionManager


class TestSession(unittest.TestCase):
    def test_add_and_get_history(self):
        s = Session("sid-1")
        s.add_message("user", "你好")
        s.add_message("assistant", "你好，有什么可以帮你")
        hist = s.get_history()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0], {"role": "user", "content": "你好"})

    def test_get_history_returns_copy(self):
        s = Session("sid-1")
        s.add_message("user", "hi")
        hist = s.get_history()
        hist.append({"role": "user", "content": "外部修改"})
        # 内部状态不应被外部修改影响
        self.assertEqual(len(s.messages), 1)

    def test_clear(self):
        s = Session("sid-1")
        s.add_message("user", "hi")
        s.clear()
        self.assertEqual(s.get_history(), [])


class TestSessionManager(unittest.TestCase):
    def test_create_session(self):
        m = SessionManager()
        s = m.create_session()
        self.assertTrue(s.session_id)
        self.assertIn(s.session_id, m.list_sessions())

    def test_create_duplicate_raises(self):
        m = SessionManager()
        m.create_session(session_id="abc")
        with self.assertRaises(ValueError):
            m.create_session(session_id="abc")

    def test_get_session(self):
        m = SessionManager()
        m.create_session(session_id="abc")
        self.assertEqual(m.get_session("abc").session_id, "abc")

    def test_get_missing_raises(self):
        m = SessionManager()
        with self.assertRaises(KeyError):
            m.get_session("nonexist")

    def test_get_or_create(self):
        m = SessionManager()
        s1 = m.get_or_create("u1")
        s2 = m.get_or_create("u1")
        self.assertIs(s1, s2)  # 同一个 session
        self.assertEqual(len(m), 1)

    def test_destroy(self):
        m = SessionManager()
        m.create_session(session_id="abc")
        self.assertTrue(m.destroy_session("abc"))
        self.assertFalse(m.destroy_session("abc"))  # 已销毁
        self.assertNotIn("abc", m.list_sessions())

    def test_isolation_between_sessions(self):
        m = SessionManager()
        a = m.create_session(session_id="A")
        b = m.create_session(session_id="B")
        a.add_message("user", "A 的问题")
        b.add_message("user", "B 的问题")
        # 两个 session 的历史完全隔离
        self.assertEqual(len(a.messages), 1)
        self.assertEqual(len(b.messages), 1)
        self.assertNotEqual(a.messages[0]["content"], b.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
