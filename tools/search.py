"""search 工具：模拟搜索 API（MOCK）。

返回预设的模拟数据，仅用于演示与测试，不发起真实网络请求。
"""

from typing import Any, List

from tool_registry import ToolRegistry, tool

# 预设模拟数据：keyword -> 搜索结果列表
_MOCK_DATA: dict = {
    "python": [
        {"title": "Python 官方文档", "url": "https://docs.python.org/3/"},
        {"title": "Python 教程 - 菜鸟教程", "url": "https://www.runoob.com/python3/"},
    ],
    "agent": [
        {"title": "什么是 AI Agent", "url": "https://example.com/ai-agent"},
        {"title": "ReAct 论文", "url": "https://arxiv.org/abs/2210.03629"},
    ],
    "deepseek": [
        {"title": "DeepSeek 开放平台", "url": "https://platform.deepseek.com/"},
    ],
}

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词",
        }
    },
    "required": ["query"],
}


def register(registry: ToolRegistry) -> None:
    """将 search 工具注册到给定注册表。"""

    @tool(
        registry,
        name="search",
        description="（MOCK）模拟搜索 API，根据关键词返回预设的模拟搜索结果，不发起真实网络请求",
        parameters=SEARCH_SCHEMA,
    )
    def search(query: str) -> List[dict]:
        """按关键词返回预设模拟搜索结果。"""
        # 大小写不敏感匹配；无命中时返回空列表
        results: List[dict] = []
        for key, items in _MOCK_DATA.items():
            if key in query.lower():
                results.extend(items)
        return results
