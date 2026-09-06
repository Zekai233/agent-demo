"""todo 工具：代办事项 CRUD 操作（添加、查看、删除）。"""

from typing import Any, Dict, List

from tool_registry import ToolRegistry, tool

# 模块级存储：待办事项列表（内存态，进程内有效）
_todos: List[Dict[str, Any]] = []
_next_id = 1


def _reset() -> None:
    """重置存储（供测试使用）。"""
    global _todos, _next_id
    _todos = []
    _next_id = 1


def _get_all() -> List[Dict[str, Any]]:
    return list(_todos)


# --- 三个动作共享的工具名不同，分别注册 ---

ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "待办事项内容",
        }
    },
    "required": ["content"],
}

LIST_SCHEMA = {"type": "object", "properties": {}}

DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "integer",
            "description": "要删除的待办事项 ID",
        }
    },
    "required": ["id"],
}


def register(registry: ToolRegistry) -> None:
    """将 todo 相关工具注册到给定注册表。"""

    @tool(
        registry,
        name="todo_add",
        description="添加一条待办事项",
        parameters=ADD_SCHEMA,
    )
    def todo_add(content: str) -> Dict[str, Any]:
        global _next_id
        item = {"id": _next_id, "content": content, "done": False}
        _todos.append(item)
        _next_id += 1
        return item

    @tool(
        registry,
        name="todo_list",
        description="查看所有待办事项",
        parameters=LIST_SCHEMA,
    )
    def todo_list() -> List[Dict[str, Any]]:
        return _get_all()

    @tool(
        registry,
        name="todo_delete",
        description="按 ID 删除一条待办事项",
        parameters=DELETE_SCHEMA,
    )
    def todo_delete(id: int) -> Dict[str, Any]:
        for i, item in enumerate(_todos):
            if item["id"] == id:
                removed = _todos.pop(i)
                return {"deleted": removed, "remaining": _get_all()}
        raise ValueError(f"未找到 ID 为 {id} 的待办事项")
