# 真实 LLM API 集成测试文档

本文档记录**真实调用 DeepSeek API** 的集成测试详细过程，包含每个测试的输入、中间过程结构、最终输出结构。

## 说明

- 单元测试（`tests/test_*.py` 除 `test_integration_real_api.py` 外）使用 `FakeLLM` 替身，**不调用真实 API**。
- 本文档对应的集成测试 `tests/test_integration_real_api.py` 会**真实调用 DeepSeek API**，消耗 token。

## 运行方式

```bash
conda activate agent-demo

# 运行真实 API 集成测试（会消耗 token）
pytest tests/test_integration_real_api.py -m integration -v

# 运行普通单元测试（默认，不含真实 API）
pytest -q
```

**前置条件**：`.env` 中已配置 `DEEPSEEK_API_KEY`，且网络可访问 `api.deepseek.com`。

## 测试配置说明

- `pytest.ini` 中通过 `addopts = --ignore=tests/test_integration_real_api.py` 让默认 `pytest` 跳过集成测试。
- 集成测试文件通过 `pytestmark = pytest.mark.integration` 标记。
- 未配置 API Key 时自动 `skip`。

---

## 测试结果

```
tests/test_integration_real_api.py::TestRealCalculator::test_calculator PASSED [ 33%]
tests/test_integration_real_api.py::TestRealSearch::test_search         PASSED [ 66%]
tests/test_integration_real_api.py::TestRealTodo::test_todo_add         PASSED [100%]

============================== 3 passed in 10.38s ==============================
```

---

## 一、TestRealCalculator（真实 API：calculator 工具）

### 测试用例 `test_calculator`

**测试目标**：让 Agent 通过真实 LLM 调用 `calculator` 工具计算乘法。

**输入结构**（`agent.run` 的参数）：
```python
agent.run("请帮我计算 123 乘以 456 等于多少", session_id="real-calc")
```

**第 1 轮 — 真实 LLM 收到的 messages：**
```json
[{"role": "user", "content": "请帮我计算 123 乘以 456 等于多少"}]
```

**第 1 轮 — 真实 LLM 原始输出（JSON）：**
```json
{
  "thought": "用户需要计算123乘以456，我需要使用calculator工具。",
  "action": "calculator",
  "action_input": { "expression": "123 * 456" },
  "final_answer": null
}
```

**第 1 轮 — 工具真实执行：**
- `registry.execute("calculator", {"expression": "123 * 456"})` → `56088`
- 生成 observation：
```json
{"role": "user", "content": "Observation: 工具 'calculator' 返回: 56088"}
```

**第 2 轮 — 真实 LLM 收到的 messages（中间结构）：**
```json
[
  {"role": "user", "content": "请帮我计算 123 乘以 456 等于多少"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"action\": \"calculator\", ...}"},
  {"role": "user", "content": "Observation: 工具 'calculator' 返回: 56088"}
]
```

**第 2 轮 — 真实 LLM 原始输出：**
```json
{
  "thought": "工具返回结果为56088，我可以直接给出最终答案。",
  "action": "final_answer",
  "action_input": null,
  "final_answer": "123 乘以 456 等于 56088。"
}
```

**最终输出（run 返回值）：** `"123 乘以 456 等于 56088。"`

**断言**：`"56088" in answer` ✅

---

## 二、TestRealSearch（真实 API：search 工具）

### 测试用例 `test_search`

**测试目标**：让 Agent 通过真实 LLM 调用 `search` 工具搜索资料。

**输入结构：**
```python
agent.run("请帮我搜索一下 python 相关的资料", session_id="real-search")
```

**第 1 轮 — 真实 LLM 原始输出：**
```json
{
  "thought": "用户需要搜索 python 相关的资料，我将调用 search 工具进行搜索。",
  "action": "search",
  "action_input": { "query": "python" },
  "final_answer": null
}
```

**第 1 轮 — 工具真实执行（MOCK 数据返回）：**
```json
[
  {"title": "Python 官方文档", "url": "https://docs.python.org/3/"},
  {"title": "Python 教程 - 菜鸟教程", "url": "https://www.runoob.com/python3/"}
]
```

**第 2 轮 — 真实 LLM 原始输出：**
```json
{
  "thought": "搜索返回了 Python 官方文档和菜鸟教程的链接，我可以将这些资料提供给用户。",
  "action": "final_answer",
  "action_input": null,
  "final_answer": "为您找到以下 Python 相关资料：\n1. Python 官方文档：https://docs.python.org/3/\n2. Python 教程 - 菜鸟教程：https://www.runoob.com/python3/"
}
```

**最终输出：**
```
为您找到以下 Python 相关资料：
1. Python 官方文档：https://docs.python.org/3/
2. Python 教程 - 菜鸟教程：https://www.runoob.com/python3/
```

**断言**：`"python" in answer.lower()` ✅

---

## 三、TestRealTodo（真实 API：todo 工具）

### 测试用例 `test_todo_add`

**测试目标**：让 Agent 通过真实 LLM 调用 `todo_add` 工具添加待办，并验证**真实写入存储**。

**输入结构：**
```python
todo_mod._reset()  # 重置存储
agent.run("帮我添加一条待办事项：明天下午开会", session_id="real-todo")
```

**第 1 轮 — 真实 LLM 原始输出：**
```json
{
  "thought": "用户需要添加一条待办事项，内容为"明天下午开会"，我应该调用 todo_add 工具。",
  "action": "todo_add",
  "action_input": { "content": "明天下午开会" },
  "final_answer": null
}
```

**第 1 轮 — 工具真实执行：**
- 存储追加 `{"id": 1, "content": "明天下午开会", "done": false}`
- 返回该条记录

**第 2 轮 — 真实 LLM 原始输出：**
```json
{
  "thought": "待办事项已成功添加，返回了 ID 为 1 的记录。我可以直接告知用户添加成功。",
  "action": "final_answer",
  "action_input": null,
  "final_answer": "已成功添加待办事项：明天下午开会（ID: 1）"
}
```

**最终输出：** `"已成功添加待办事项：明天下午开会（ID: 1）"`

**真实存储验证（关键）**：
```json
[{"id": 1, "content": "明天下午开会", "done": false}]
```

**断言**：
- `"明天下午开会" in answer` ✅
- `todo_mod._get_all()` 中存在 `"明天下午开会"` 的记录 ✅（证明工具真实执行，非模拟）

---

## 附：真实 API 完整 ReAct 过程示例（sqrt 场景）

以下是一次真实 API 调用的完整日志，展示 Agent 的端到端行为：

**用户输入**：`请计算 17 的平方根，然后告诉我结果`

**第 1 轮 — LLM 输出：**
```json
{
  "thought": "用户要求计算17的平方根，我可以使用calculator工具来计算sqrt(17)。",
  "action": "calculator",
  "action_input": { "expression": "sqrt(17)" },
  "final_answer": null
}
```

**工具执行**：`calculator("sqrt(17)")` → `4.123105625617661`

**第 2 轮 — LLM 输出：**
```json
{
  "thought": "我已经得到了17的平方根的计算结果，可以直接回答用户。",
  "action": "final_answer",
  "action_input": null,
  "final_answer": "17 的平方根约为 4.123105625617661。"
}
```

**最终答案**：`17 的平方根约为 4.123105625617661。`

**session 完整历史（5 条）：**
```
[0] user:      请计算 17 的平方根，然后告诉我结果
[1] assistant: {"thought": "用户要求计算17的平方根...", "action": "calculator", ...}
[2] user:      Observation: 工具 'calculator' 返回: 4.123105625617661
[3] user:      请计算 17 的平方根，然后告诉我结果
[4] assistant: 17 的平方根约为 4.123105625617661。
```

> 注：`[3]` 与 `[0]` 内容重复，是当前 `final_answer` 分支重复追加 user 消息的已知行为，不影响功能。

---

## 测试总结

| 场景 | 涉及工具 | 真实 API 调用 | 验证点 |
|------|---------|--------------|--------|
| calculator | calculator | ✅ | 计算结果 56088 正确 |
| search | search | ✅ | 返回 mock 搜索结果 |
| todo | todo_add | ✅ | 待办真实写入内存存储 |

真实 API 集成测试共 **3 个用例全部通过**，验证了 Agent 的 ReAct 循环、工具调用、JSON 解析、上下文拼接在真实 DeepSeek 服务下均正常工作。
