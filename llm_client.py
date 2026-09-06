"""DeepSeek LLM 客户端封装。

提供统一的聊天接口，支持自定义 System Prompt 与历史消息，
仅返回模型生成的原始文本，供上层 Agent 后续解析使用。
"""

import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

# 从项目根目录的 .env 文件加载环境变量（DEEPSEEK_API_KEY）
load_dotenv()


class LLMClient:
    """基于 OpenAI 兼容协议封装的 DeepSeek 客户端。"""

    # DeepSeek 官方 API 地址
    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        参数:
            api_key: DeepSeek API Key。若为空，则尝试从环境变量
                     DEEPSEEK_API_KEY 读取。
                     【填写 API Key 的位置见本文件底部 __main__ 示例】
            model: 模型名称，默认 deepseek-chat。
            temperature: 采样温度，控制随机性 (0~2)。
            max_tokens: 单次回复的最大 token 数。
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "缺少 API Key。请通过参数 api_key 传入，"
                "或设置环境变量 DEEPSEEK_API_KEY。"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> str:
        """发送对话并返回模型生成的原始文本。

        参数:
            messages: 历史/上下文消息列表，每项形如
                      {"role": "user" | "assistant", "content": "..."}
            system_prompt: 系统提示词，可选。

        返回:
            str: 模型回复的原始文本，供后续解析。
        """
        if system_prompt:
            full_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
        else:
            full_messages = list(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 仅返回文本内容
        return response.choices[0].message.content


if __name__ == "__main__":
    # ============================================================
    # API Key 从 .env 文件的 DEEPSEEK_API_KEY 读取，
    # 无需在此处填写。首次使用请：cp .env.example .env 后填入 Key。
    # ============================================================
    client = LLMClient()

    reply = client.chat(
        system_prompt="你是一个乐于助人的助手。",
        messages=[{"role": "user", "content": "你好，请做一下自我介绍。"}],
    )
    print(reply)
