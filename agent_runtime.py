"""核心 Agent 运行时：基于 ReAct 模型的 Agent 执行循环。

流程：思考(Thought) -> 行动(Action) -> 观察(Observation)，循环直至产出 final_answer。
LLM 被要求强制输出 JSON，包含 thought / action / action_input / final_answer 字段。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from llm_client import LLMClient
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
    ) -> None:
        """
        参数:
            llm: LLM 客户端实例。
            registry: 工具注册表。
            max_iterations: 最大循环次数，防止死循环。
        """
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tools_schema = json.dumps(
            self.registry.get_all_tools_schema(), ensure_ascii=False, indent=2
        )
        return SYSTEM_PROMPT_TEMPLATE.format(tools_schema=tools_schema)

    def run(self, user_query: str) -> str:
        """执行 ReAct 循环，返回最终答案。

        参数:
            user_query: 用户输入。

        返回:
            str: 最终答案文本。
        """
        system_prompt = self._build_system_prompt()
        # messages 仅保留 user 初始输入，后续通过 observation 追加 assistant 消息来推进
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": user_query}
        ]

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[Agent] 第 {iteration} 轮思考")

            raw = self.llm.chat(messages=messages, system_prompt=system_prompt)
            logger.info(f"[Agent] LLM 原始输出: {raw}")

            parsed = self._parse_output(raw)

            # 将本轮 LLM 输出作为 assistant 消息追加，保持上下文连贯
            messages.append({"role": "assistant", "content": raw})

            action = parsed.get("action", "")
            thought = parsed.get("thought", "")
            action_input = parsed.get("action_input")
            final_answer = parsed.get("final_answer")

            if action == "final_answer":
                logger.info("[Agent] 产出最终答案")
                return final_answer or ""

            # 工具调用分支
            if action:
                try:
                    result = self._execute_tool(action, action_input)
                    observation = f"工具 '{action}' 返回: {json.dumps(result, ensure_ascii=False, default=str)}"
                except Exception as exc:  # noqa: BLE001 工具报错需反馈给 LLM
                    observation = f"工具 '{action}' 执行出错: {type(exc).__name__}: {exc}"
                    logger.error(f"[Agent] 工具调用失败: {observation}")
                # 将观察结果作为 user 消息拼接到上下文
                messages.append({"role": "user", "content": f"Observation: {observation}"})
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

        # 达到最大循环次数仍未结束
        logger.warning(f"[Agent] 达到最大循环次数 {self.max_iterations}，强制终止")
        return "抱歉，我在处理你的问题时超过了最大循环次数，请稍后再试。"

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
