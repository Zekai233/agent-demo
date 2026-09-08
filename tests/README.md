# Agent 单元测试文档（详细版）

本文档详细记录基于 pytest 的每个单元测试用例的**输入结构、中间过程结构、最终输出结构**。

## 测试环境

| 项目 | 值 |
|------|-----|
| Python | 3.11.13（conda 环境 `agent-demo`） |
| pytest | 9.1.1 |
| 测试目录 | `tests/` |
| 测试结果 | **41 passed**（全量通过） |

## 运行方式

```bash
conda activate agent-demo
pytest tests/ -v
```

> 注意：`pytest` 默认运行**单元测试**（使用 `FakeLLM` 替身，不调用真实 API）。
> 另有**真实 LLM API 集成测试**，见 [README_integration.md](./README_integration.md)，
> 需显式运行 `pytest tests/test_integration_real_api.py -m integration -v`。

## 测试基础设施

### FakeLLM（LLM 替身）

`tests/conftest.py` 中的 `FakeLLM` 是可编程的 LLM 替身，用于替代真实 DeepSeek API：

```python
class FakeLLM:
    def __init__(self, responses):   # responses: 预设的响应列表
        self.responses = list(responses)
        self.calls = []               # 记录每次 chat() 的调用参数

    def chat(self, messages, system_prompt=""):
        self.calls.append({"messages": list(messages), "system_prompt": system_prompt})
        resp = self.responses.pop(0)  # 按顺序弹出响应
        if isinstance(resp, Exception):
            raise resp                # 支持抛异常模拟失败
        return resp
```

**用途**：`llm.calls[i]["messages"]` 可断言第 i 次 LLM 调用时收到的完整上下文结构。

---

## 一、工具注册测试（tests/test_tools.py）

### 1.1 `test_register_and_schema`

**测试目标**：注册工具并导出 schema。

**输入结构**（装饰器参数）：
```python
@tool(registry, name="add", description="计算两个数的和",
      parameters={
          "type": "object",
          "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
          "required": ["a", "b"],
      })
def add(a, b):
    return a + b
```

**中间结构**（`registry._tools` 内部存储）：
```python
{
  "add": {
    "name": "add",
    "description": "计算两个数的和",
    "parameters": {...},   # 上面的 JSON Schema
    "func": <function add>,
  }
}
```

**输出结构**（`get_all_tools_schema()`）：
```json
[
  {
    "type": "function",
    "function": {
      "name": "add",
      "description": "计算两个数的和",
      "parameters": {
        "type": "object",
        "properties": { "a": {"type": "number"}, "b": {"type": "number"} },
        "required": ["a", "b"]
      }
    }
  }
]
```

### 1.2 `test_duplicate_register_raises`

**输入**：对同名工具 `foo` 重复注册两次。

**输出**：抛出 `ValueError("工具 'foo' 已存在，请勿重复注册")`。

### 1.3 `test_execute_unknown_raises`

**输入**：`registry.execute("nonexistent", {})`。

**输出**：抛出 `KeyError("工具 'nonexistent' 未注册")`。

### 1.4 `test_register_all_tools`

**输入**：`register_all_tools(registry)`。

**输出**（5 个工具名集合）：
```
{"calculator", "search", "todo_add", "todo_list", "todo_delete"}
```

---

## 二、具象工具功能测试（tests/test_tools.py）

### 2.1 Calculator（计算器）

#### `test_basic`
- **输入**：`registry.execute("calculator", {"expression": "3 + 4 * 2"})`
- **中间结构**：asteval 解析表达式，符号表求值
- **输出**：`11`

#### `test_math_function`
- **输入**：`{"expression": "sqrt(16)"}`
- **输出**：`4`

#### `test_power`
- **输入**：`{"expression": "2 ** 10"}`
- **输出**：`1024`

#### `test_invalid_expression`
- **输入**：`{"expression": "1 / 0"}`（除零）
- **输出**：抛出异常（`asteval` 检测到除零错误，转为 `ValueError`）

#### `test_no_dangerous_eval`
- **输入**：`{"expression": "__import__('os').system('ls')"}`（恶意代码）
- **输出**：抛出异常（`minimal=True` 白名单拦截，无系统访问能力）

### 2.2 Search（模拟搜索，MOCK）

#### `test_hit`
- **输入**：`registry.execute("search", {"query": "python"})`
- **中间结构**：遍历 `_MOCK_DATA`，匹配 `key in query.lower()`
- **输出**：
```json
[
  {"title": "Python 官方文档", "url": "https://docs.python.org/3/"},
  {"title": "Python 教程 - 菜鸟教程", "url": "https://www.runoob.com/python3/"}
]
```

#### `test_miss`
- **输入**：`{"query": "不存在的词xyz"}`
- **输出**：`[]`（空列表）

#### `test_case_insensitive`
- **输入**：`{"query": "PYTHON"}`（大写）
- **输出**：命中 2 条（`query.lower()` 做大小写不敏感匹配）

### 2.3 Todo（待办事项 CRUD）

#### `test_add`
- **输入**：`registry.execute("todo_add", {"content": "写周报"})`
- **中间结构**（内部存储追加）：
```python
_todos = [{"id": 1, "content": "写周报", "done": False}]
_next_id = 2
```
- **输出**：
```json
{"id": 1, "content": "写周报", "done": false}
```

#### `test_list`
- **输入**：先 add "任务A"、add "任务B"，再 `registry.execute("todo_list", {})`
- **输出**：
```json
[
  {"id": 1, "content": "任务A", "done": false},
  {"id": 2, "content": "任务B", "done": false}
]
```

#### `test_delete`
- **输入**：先 add "任务A"，再 `registry.execute("todo_delete", {"id": 1})`
- **输出**：
```json
{
  "deleted": {"id": 1, "content": "任务A", "done": false},
  "remaining": []
}
```

#### `test_delete_not_found`
- **输入**：`registry.execute("todo_delete", {"id": 999})`
- **输出**：抛出 `ValueError("未找到 ID 为 999 的待办事项")`

---

## 三、主循环解析与异常处理测试（tests/test_agent_runtime.py）

### 3.1 解析逻辑（`_parse_output`）

#### `test_plain_json`
- **输入**：`'{"thought": "t", "action": "final_answer", "final_answer": "ok"}'`
- **输出**：`{"thought": "t", "action": "final_answer", "final_answer": "ok"}`

#### `test_json_with_code_block`
- **输入**：`` ```json\n{"thought": "t", ...}\n``` ``
- **中间结构**：剥离 ```` ```json ```` 代码块标记
- **输出**：解析出的 dict

#### `test_extract_json_from_extra_text`
- **输入**：`前面有文字 {"thought": "t", ...} 后面有文字`
- **中间结构**：`find("{")` + `rfind("}")` 提取首个 `{...}` 片段
- **输出**：解析出的 dict

#### `test_invalid_json_raises`
- **输入**：`"这不是 JSON"`
- **输出**：抛出 `ValueError("无法解析 LLM 输出为 JSON: ...")`

### 3.2 主循环

#### `test_final_answer_directly`
- **输入**：`agent.run("打招呼")`，LLM 返回 `{"action": "final_answer", "final_answer": "你好"}`
- **中间结构**（第 1 次 LLM 调用的 messages）：
```json
[{"role": "user", "content": "打招呼"}]
```
- **输出**：`"你好"`

#### `test_tool_then_final_answer`（完整工具调用流程）

**输入**：`agent.run("帮我算 3+4")`

**第 1 轮 — LLM 输入 messages：**
```json
[{"role": "user", "content": "帮我算 3+4"}]
```

**第 1 轮 — LLM 原始输出：**
```json
{"thought": "需要计算", "action": "calculator", "action_input": {"expression": "3+4"}, "final_answer": null}
```

**第 1 轮 — 工具执行：**
- `registry.execute("calculator", {"expression": "3+4"})` → `7`
- 生成 observation 并追加：
```json
{"role": "user", "content": "Observation: 工具 'calculator' 返回: 7"}
```

**第 2 轮 — LLM 输入 messages（关键中间结构）：**
```json
[
  {"role": "user", "content": "帮我算 3+4"},
  {"role": "assistant", "content": "{\"thought\": \"需要计算\", \"action\": \"calculator\", ...}"},
  {"role": "user", "content": "Observation: 工具 'calculator' 返回: 7"}
]
```

**第 2 轮 — LLM 输出：**
```json
{"thought": "得到结果", "action": "final_answer", "action_input": null, "final_answer": "结果是7"}
```

**最终输出（run 返回值）：** `"结果是7"`

### 3.3 工具异常处理（可恢复错误）

#### `test_tool_error_prefix`
- **输入**：`agent.run("计算", session_id="s1")`，LLM 让 calculator 计算 `"1/0"`
- **中间结构**（工具抛 ValueError 后转成 observation）：
```json
{"role": "user", "content": "Observation: [Tool Error] 工具 'calculator' 执行出错: ValueError: 表达式解析失败: ..."}
```
- **中间结构**（写回 session 后）：
```json
[
  {"role": "user", "content": "计算"},
  {"role": "assistant", "content": "{...calculator 调用...}"},
  {"role": "user", "content": "Observation: [Tool Error] 工具 'calculator' 执行出错: ValueError: ..."}
]
```
- **输出**：`"请重新输入"`（LLM 收到错误后纠正）

#### `test_tool_error_not_propagated`
- **输入**：LLM 调用 `todo_delete({"id": 999})` 抛 ValueError
- **验证**：异常被捕获，`run()` 正常返回，不向外抛出

#### `test_consecutive_errors_inject_apology`
- **输入**：LLM 连续两次调用 `todo_delete({"id": 999})` 均报错
- **中间结构**（第 2 次报错后注入）：
```json
{"role": "user", "content": "Observation: 由于工具连续报错，请向用户致歉并请求换一种方式提问，不要尝试调用该工具了。"}
```
- **输出**：`"抱歉，请换个方式"`（LLM 基于致歉提示生成的最终答复）

### 3.4 致命异常处理（不可恢复错误）

#### `test_llm_failure_fallback`
- **输入**：LLM 抛 `Exception("connection timeout")`
- **中间结构**：`run()` 内部 catch，包装为 `FatalError`
- **输出**：`"抱歉，系统暂时无法连接服务器，请稍后再试。"`

#### `test_llm_failure_writes_system_error`
- **输入**：同上（LLM 抛超时）
- **中间结构**（写回 session 的消息，不含原始错误细节）：
```json
[{"role": "system", "content": "System Error: 处理过程中发生不可恢复错误"}]
```
- **验证**：session 中 `role == "system"`，且内容不包含 `"connection timeout"`

#### `test_parse_error_fallback`
- **输入**：LLM 返回 `"完全无法解析的乱码###"`
- **中间结构**：`_parse_output` 抛出 ValueError → 包装为 `FatalError`
- **输出**：`"抱歉，模型暂时无法理解该问题，请换个方式描述后再试。"`

---

## 四、Session 隔离性测试（tests/test_session.py）

### 4.1 `test_add_and_get`
- **输入**：`Session("sid")`，`add_message("user", "你好")`、`add_message("assistant", "你好呀")`
- **输出**：`get_history()` 返回 2 条消息

### 4.2 `test_get_history_returns_copy`
- **输入**：add 1 条消息，`hist = get_history()`，外部 `hist.append(...)`
- **输出**：`s.messages` 仍为 1 条（返回的是副本，外部修改无效）

### 4.3 `test_create_get_destroy`
- **输入**：`create_session(session_id="abc")` → `get_session("abc")` → `destroy_session("abc")`
- **输出**：销毁返回 `True`，二次销毁返回 `False`

### 4.4 `test_get_or_create`
- **输入**：两次 `get_or_create("u1")`
- **输出**：两次返回同一实例（`s1 is s2`）

### 4.5 `test_isolation_between_sessions`
- **输入**：Session A add `"A 的问题"`，Session B add `"B 的问题"`
- **中间结构**：
```python
a.messages = [{"role": "user", "content": "A 的问题"}]
b.messages = [{"role": "user", "content": "B 的问题"}]
```
- **输出**：断言 `a.messages[0]["content"] != b.messages[0]["content"]`

### 4.6 `test_runtime_multi_session_isolated`
- **输入**：`agent.run("A的问题", session_id="A")`、`agent.run("B的问题", session_id="B")`
- **中间结构**（两个 session 各自历史）：
```python
A 的 session: [{"role": "user", "content": "A的问题"}, {"role": "assistant", "content": "A答"}]
B 的 session: [{"role": "user", "content": "B的问题"}, {"role": "assistant", "content": "B答"}]
```
- **输出**：两个 session 历史互不相同

### 4.7 `test_runtime_second_call_loads_history`
- **输入**：同 session "u1" 连续 `run("问题1")`、`run("问题2")`
- **中间结构**（第 2 次 run 时 LLM 收到的 messages）：
```json
[
  {"role": "user", "content": "问题1"},
  {"role": "assistant", "content": "第一答"},
  {"role": "user", "content": "问题2"}
]
```
- **输出**：验证历史 `"问题1"` 和本轮 `"问题2"` 都在上下文中

---

## 五、连续对话与 context 压缩测试（tests/test_context.py）

### 5.1 `test_compress_triggers_when_long`
- **输入**（超长历史，4 条各 1000 字符）：
```python
[
  {"role": "user", "content": "x" * 1000},
  {"role": "assistant", "content": "y" * 1000},
  {"role": "user", "content": "z" * 1000},
  {"role": "assistant", "content": "w" * 1000},
]
```
- **中间结构**（触发判定）：总字符数 4000 > 阈值 100
- **中间结构**（LLM 摘要调用）：把前 2 条旧历史交给 LLM，返回 `"这是历史摘要"`
- **输出**（压缩后 3 条，`keep_recent_messages=2`）：
```json
[
  {"role": "system", "content": "以下是之前对话的摘要：\n这是历史摘要"},
  {"role": "user", "content": "zzz..."},
  {"role": "assistant", "content": "www..."}
]
```

### 5.2 `test_compress_not_triggered_when_short`
- **输入**（短历史 2 条）：
```python
[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
```
- **输出**：原样返回，不触发压缩

### 5.3 `test_compress_keeps_tool_results`
- **输入**（历史含工具结果）：
```python
[
  {"role": "user", "content": "旧问题" * 500},
  {"role": "assistant", "content": "旧回答" * 500},
  {"role": "user", "content": "Observation: 工具 'calculator' 返回: 42"},
  {"role": "assistant", "content": "{...final_answer...}"},
]
```
- **输出**：压缩后 `contents` 中仍包含 `"工具 'calculator' 返回: 42"`（最近消息被保留）

### 5.4 `test_compress_summary_failure_falls_back`
- **输入**：超长历史 + LLM 摘要抛 `Exception("摘要服务不可用")`
- **中间结构**：`_summarize` 捕获异常，退化为 `history_text[-500:]` 截断
- **输出**：压缩后最近 2 条消息仍保留，流程不中断

### 5.5 `test_tool_result_in_followup`（带工具追问）

**输入**：同 session "s1" 先 `run("3+4")` 再 `run("刚才结果是多少")`

**第 1 次 run 中间结构**（工具结果写入 session）：
```json
{"role": "user", "content": "Observation: 工具 'calculator' 返回: 7"}
```

**第 2 次 run 时 LLM 收到的 messages（关键验证点）：**
```json
[
  {"role": "user", "content": "3+4"},
  {"role": "assistant", "content": "{...calculator 调用...}"},
  {"role": "user", "content": "Observation: 工具 'calculator' 返回: 7"},
  {"role": "assistant", "content": "结果是7"},
  {"role": "user", "content": "刚才结果是多少"}
]
```
- **输出**：验证第 3 次 LLM 调用上下文包含 `"工具 'calculator' 返回: 7"`

### 5.6 `test_compress_triggered_in_long_conversation`
- **输入**：`agent.run("计算", session_id="s1")`，`max_context_chars=10`
- **中间结构**：第 1 轮调用 calculator → 第 2 轮前触发压缩（`_summarize` 消耗 1 个响应）
- **输出**：`"完成"`，且验证某次调用 messages 中存在 `role == "system"` 且含 `"摘要"` 的消息

---

## 测试结果汇总

```
============================== 41 passed in 2.22s ==============================
```

| 测试文件 | 用例数 | 覆盖内容 |
|---------|-------|---------|
| test_tools.py | 15 | 工具注册 + calculator/search/todo |
| test_agent_runtime.py | 12 | 解析逻辑 + 可恢复/不可恢复异常 |
| test_session.py | 7 | Session 隔离性 |
| test_context.py | 7 | context 压缩 + 连续对话 |
| **合计** | **41** | |
