"""核心 Agent 运行时：基于 ReAct 模型的 Agent 执行循环。

流程：思考(Thought) -> 行动(Action) -> 观察(Observation)，循环直至产出 final_answer。
LLM 被要求强制输出 JSON，包含 thought / action / action_input / final_answer 字段。

异常处理分两类：
- 可恢复错误（工具执行异常，如非法公式、找不到 id）：
    捕获后转为 "[Tool Error]" 前缀的 observation，追加到 session，让 LLM 自行纠正。
- 不可恢复错误（LLM API 失效、网络超时、输出乱码）：
    由 Runtime 层直接 catch，向 Session 写入内部 System Error 消息，
    向用户返回友好降级提示并终止循环，不把原始错误信息粗暴塞入上下文。
"""

import json
import logging
from typing import Any, Dict, Optional

from llm_client import LLMClient
from session import Session, SessionManager
from tool_registry import ToolRegistry

# trace 日志：记录每次工具调用的输入输出
logger = logging.getLogger("agent_runtime")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(_handler)


class ToolError(Exception):
    """可恢复的工具执行错误：应由 LLM 收到反馈后自行纠正。"""


class FatalError(Exception):
    """不可恢复的致命错误：应终止循环并向用户返回降级提示。"""


# 同一工具连续报错达到该次数时，注入致歉提示并终止
MAX_CONSECUTIVE_TOOL_ERRORS = 2

# 友好降级提示文案
_FALLBACK_MESSAGES = {
    "llm": "抱歉，系统暂时无法连接服务器，请稍后再试。",
    "parse": "抱歉，模型暂时无法理解该问题，请换个方式描述后再试。",
    "unknown": "抱歉，系统出现内部错误，请稍后再试。",
}


# ReAct 系统提示词模板，要求 LLM 强制输出 JSON
SYSTEM_PROMPT_TEMPLATE = """你是一个能够使用工具的智能体(Agent)，请严格遵循 ReAct 模式解决问题。

你必须始终只输出一个合法的 JSON 对象，不要输出任何其他文字、代码块标记或解释。

可用工具列表（JSON Schema 格式）：
{tools_schema}

输出 JSON 的字段约定：
- "thought": 字符串，你的思考过程。
- "action": 字符串，当需要调用工具时，为要调用的工具名；当可以直接回答时，为 "final_answer"。
- "action_input": 对象，调用工具时传入的参数（JSON 对象）；若 action 为 "final_answer"，可为 null。
- "final_answer": 字符串，最终回答；仅当 action 为 "final_answer" 时填写，否则为 null。

规则：
1. 每次只能调用一个工具，调用后你会收到 observation 结果，再继续思考。
2. 当你已经能给出最终答案时，action 必须为 "final_answer"，并在 final_answer 中给出答案。
3. 不要臆造工具结果，所有信息必须来自工具返回的 observation。
"""


class AgentRuntime:
    """ReAct Agent 运行时。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_iterations: int = 10,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        """
        参数:
            llm: LLM 客户端实例。
            registry: 工具注册表。
            max_iterations: 最大循环次数，防止死循环。
            session_manager: 会话管理器；缺省时自动创建，用于多用户窗口隔离。
        """
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self.session_manager = session_manager or SessionManager()

    def _build_system_prompt(self) -> str:
        tools_schema = json.dumps(
            self.registry.get_all_tools_schema(), ensure_ascii=False, indent=2
        )
        return SYSTEM_PROMPT_TEMPLATE.format(tools_schema=tools_schema)

    def run(self, user_query: str, session_id: Optional[str] = None) -> str:
        """执行 ReAct 循环，返回最终答案。

        参数:
            user_query: 用户输入。
            session_id: 可选，会话 ID。传入时从对应 Session 读取历史上下文，
                        并将本轮对话写回该 Session，实现多用户窗口隔离；
                        缺省时不持久化（每次独立对话）。

        返回:
            str: 最终答案文本（含降级提示）。
        """
        system_prompt = self._build_system_prompt()

        # 根据 session_id 获取对应的上下文历史
        session = None
        if session_id is not None:
            session = self.session_manager.get_or_create(session_id)
            messages = session.get_history()
            logger.info(f"[Session] 会话 '{session_id}' 已有 {len(messages)} 条历史消息")
        else:
            messages = []

        # 将本轮用户输入追加到消息列表
        messages.append({"role": "user", "content": user_query})

        # 同一工具连续报错计数器
        last_tool_name: Optional[str] = None
        consecutive_errors = 0

        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"[Agent] 第 {iteration} 轮思考")

                # --- 不可恢复错误：LLM 调用失败（API 失效/网络超时等） ---
                try:
                    raw = self.llm.chat(messages=messages, system_prompt=system_prompt)
                except Exception as exc:  # noqa: BLE001
                    raise FatalError(f"LLM 调用失败: {type(exc).__name__}: {exc}") from exc

                logger.info(f"[Agent] LLM 原始输出: {raw}")

                # --- 不可恢复错误：输出无法解析为 JSON ---
                try:
                    parsed = self._parse_output(raw)
                except Exception as exc:  # noqa: BLE001
                    raise FatalError(f"LLM 输出解析失败: {exc}") from exc

                # 将本轮 LLM 输出作为 assistant 消息追加，保持上下文连贯
                messages.append({"role": "assistant", "content": raw})

                action = parsed.get("action", "")
                action_input = parsed.get("action_input")
                final_answer = parsed.get("final_answer")

                if action == "final_answer":
                    logger.info("[Agent] 产出最终答案")
                    answer = final_answer or ""
                    if session is not None:
                        session.add_message("user", user_query)
                        session.add_message("assistant", answer)
                    return answer

                # --- 工具调用分支 ---
                if action:
                    try:
                        result = self._execute_tool(action, action_input)
                        observation = f"工具 '{action}' 返回: {json.dumps(result, ensure_ascii=False, default=str)}"
                        # 工具成功，重置连续报错计数
                        last_tool_name = None
                        consecutive_errors = 0
                    except Exception as exc:  # noqa: BLE001
                        # 可恢复错误：转为 [Tool Error] observation
                        observation = self._to_tool_error(action, exc)
                        logger.error(f"[Agent] 工具调用失败: {observation}")

                        # 记录连续报错次数
                        if action == last_tool_name:
                            consecutive_errors += 1
                        else:
                            last_tool_name = action
                            consecutive_errors = 1

                    # 将观察结果作为 user 消息拼接到上下文
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                    # 同步写回 session，确保后续追问能感知刚才的错误
                    self._sync_session(session, messages)

                    # 同一工具连续报错超阈值：注入致歉提示，让 LLM 产出最终答复
                    if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                        logger.warning(
                            f"[Agent] 工具 '{action}' 连续报错 {consecutive_errors} 次，"
                            "注入致歉提示并请求最终答复"
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Observation: 由于工具连续报错，请向用户致歉并请求"
                                    "换一种方式提问，不要尝试调用该工具了。"
                                ),
                            }
                        )
                        self._sync_session(session, messages)
                        # 让 LLM 基于该提示生成最终答复（再执行一轮）
                        continue
                else:
                    # action 为空，提示 LLM 必须给出合法 action
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Observation: 你的输出缺少合法的 action 字段，"
                                "请重新输出 JSON，且 action 必须是工具名或 'final_answer'。"
                            ),
                        }
                    )
                    self._sync_session(session, messages)

            # 达到最大循环次数仍未结束
            logger.warning(f"[Agent] 达到最大循环次数 {self.max_iterations}，强制终止")
            return "抱歉，我在处理你的问题时超过了最大循环次数，请稍后再试。"

        except FatalError as exc:
            # 不可恢复错误：Runtime 层统一降级处理
            logger.error(f"[Fatal] {exc}")
            fallback = _FALLBACK_MESSAGES.get(self._classify_fatal(exc), _FALLBACK_MESSAGES["unknown"])
            if session is not None:
                # 在 Session 中记录一条内部 System Error 消息（不塞原始错误细节）
                session.add_message("system", "System Error: 处理过程中发生不可恢复错误")
            return fallback

    @staticmethod
    def _classify_fatal(exc: FatalError) -> str:
        """根据致命错误类型返回降级提示的分类键。"""
        msg = str(exc)
        if msg.startswith("LLM 调用失败"):
            return "llm"
        if msg.startswith("LLM 输出解析失败"):
            return "parse"
        return "unknown"

    @staticmethod
    def _to_tool_error(tool_name: str, exc: Exception) -> str:
        """将工具异常转化为带 [Tool Error] 前缀的字符串。"""
        return f"[Tool Error] 工具 '{tool_name}' 执行出错: {type(exc).__name__}: {exc}"

    def _sync_session(self, session: Optional[Session], messages: list) -> None:
        """将最新 messages 状态同步写回 session。"""
        if session is not None:
            session.messages = list(messages)

    def _execute_tool(self, tool_name: str, action_input: Any) -> Any:
        """执行工具并记录 trace 日志。"""
        args = action_input if isinstance(action_input, dict) else {}
        logger.info(f"[Tool] 调用工具 '{tool_name}'，参数: {args}")
        result = self.registry.execute(tool_name, args)
        logger.info(f"[Tool] 工具 '{tool_name}' 返回: {result}")
        return result

    @staticmethod
    def _parse_output(raw: str) -> Dict[str, Any]:
        """从 LLM 原始输出中解析出 JSON 字段。

        兼容 LLM 可能包裹的 ```json ... ``` 代码块。
        """
        text = raw.strip()
        # 去除可能的代码块标记
        if text.startswith("```"):
            text = text.strip("`")
            # 去掉可能的 "json" 语言标识
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 兜底：尝试提取第一个 {...} 片段
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                raise ValueError(f"无法解析 LLM 输出为 JSON: {raw[:200]}")
        if not isinstance(data, dict):
            raise ValueError("LLM 输出必须是 JSON 对象")
        return data
