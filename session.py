"""Session 会话管理：多用户窗口隔离。

提供 Session 类（持有 session_id 与 messages 历史）与
SessionManager 类（创建、获取、销毁 session）。
"""

import uuid
from typing import Dict, List, Optional


class Session:
    """单个会话：持有唯一 ID 与对话消息历史。"""

    def __init__(self, session_id: str) -> None:
        self.session_id: str = session_id
        # 对话历史，每项形如 {"role": "user"|"assistant", "content": "..."}
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """追加一条消息到历史。"""
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """返回历史消息的副本，避免外部误改内部状态。"""
        return list(self.messages)

    def clear(self) -> None:
        """清空历史。"""
        self.messages = []


class SessionManager:
    """会话管理器：负责 Session 的创建、获取与销毁。

    不同 Session 的 messages 列表相互独立，实现多用户窗口状态隔离。
    """

    def __init__(self) -> None:
        # session_id -> Session
        self._sessions: Dict[str, Session] = {}

    def create_session(self, session_id: Optional[str] = None) -> Session:
        """创建一个新会话。

        参数:
            session_id: 可选，指定会话 ID；缺省时自动生成 UUID。

        返回:
            Session: 新建的会话对象。
        """
        if session_id is None:
            session_id = uuid.uuid4().hex
        if session_id in self._sessions:
            raise ValueError(f"会话 '{session_id}' 已存在")
        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        """按 ID 获取会话；不存在时抛出 KeyError。

        参数:
            session_id: 会话 ID。

        返回:
            Session: 对应的会话对象。
        """
        if session_id not in self._sessions:
            raise KeyError(f"会话 '{session_id}' 不存在")
        return self._sessions[session_id]

    def get_or_create(self, session_id: str) -> Session:
        """获取会话，不存在则自动创建。适合主循环按传入 session_id 获取上下文。"""
        if session_id not in self._sessions:
            return self.create_session(session_id=session_id)
        return self._sessions[session_id]

    def destroy_session(self, session_id: str) -> bool:
        """销毁指定会话。

        参数:
            session_id: 会话 ID。

        返回:
            bool: 是否成功销毁（存在则 True，否则 False）。
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """返回当前所有会话 ID 列表。"""
        return list(self._sessions.keys())

    def __len__(self) -> int:
        return len(self._sessions)
