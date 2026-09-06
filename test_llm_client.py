"""llm_client 的单元测试（不依赖真实 API，使用 mock 验证）。"""

import unittest
from unittest.mock import MagicMock, patch

from llm_client import LLMClient


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        # 用 mock 替换底层 OpenAI 客户端，避免真实网络请求
        patcher = patch("llm_client.OpenAI")
        self.mock_openai = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client

        self.llm = LLMClient(api_key="test-key")

    def test_api_key_from_env(self):
        # 验证可以从环境变量读取 Key（.env 加载后即为环境变量）
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}, clear=True):
            llm = LLMClient()
            self.assertEqual(llm.api_key, "env-key")

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                LLMClient(api_key="")

    def test_chat_with_system_prompt(self):
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="你好，我是 DeepSeek。"))]
        )

        reply = self.llm.chat(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "hi"}],
        )

        self.assertEqual(reply, "你好，我是 DeepSeek。")

        # 验证 system prompt 被置于消息列表首位
        call_args = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_args["messages"][0], {"role": "system", "content": "你是助手"})
        self.assertEqual(call_args["model"], "deepseek-chat")
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["max_tokens"], 4096)

    def test_chat_without_system_prompt(self):
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        reply = self.llm.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(reply, "ok")
        call_args = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(len(call_args["messages"]), 1)

    def test_custom_params(self):
        llm = LLMClient(api_key="k", model="deepseek-reasoner", temperature=0.1, max_tokens=100)
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        llm.chat(messages=[{"role": "user", "content": "hi"}])
        call_args = self.mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_args["model"], "deepseek-reasoner")
        self.assertEqual(call_args["temperature"], 0.1)
        self.assertEqual(call_args["max_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
