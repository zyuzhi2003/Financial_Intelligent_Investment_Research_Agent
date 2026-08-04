# 这个文件用于提供命令行入口，支持构建索引、单次问答和交互式问答。
from __future__ import annotations

from datetime import datetime
import json
import textwrap
import uuid
import typer

from src.agent import AgentRouter
from src.config import load_settings
from src.memory import HybridMemoryStore, MemoryScope
from src.mcp import MCPDispatcher
from src.mcp.nl_router import build_request_from_question
from src.rag.build_index import build_vector_index
from src.rag.retriever import has_index

app = typer.Typer(help="金融RAG+MCP智能问答系统")


def _print_structured_react_steps(react_steps: list[dict[str, str]]) -> None:
    # 这个函数用于把ReAct步骤按Thought/Action/Observation分段输出，提升可读性。
    for step in react_steps:
        step_no = str(step.get("step", "?"))
        thought = str(step.get("thought", ""))
        action = str(step.get("action", ""))
        observation = str(step.get("observation", ""))

        typer.echo(f"\n[Step {step_no}]")
        typer.echo("Thought:")
        typer.echo(textwrap.fill(thought, width=96, initial_indent="  ", subsequent_indent="  "))
        typer.echo("Action:")
        typer.echo(textwrap.fill(action, width=96, initial_indent="  ", subsequent_indent="  "))
        typer.echo("Observation:")
        for line in observation.splitlines() or [observation]:
            typer.echo(textwrap.fill(line, width=96, initial_indent="  ", subsequent_indent="  "))


def _render_route_output(output, show_react_trace: bool) -> None:
    # 这个函数用于统一输出全功能调度结果，避免ask/chat/agent-ask展示不一致。
    typer.echo("=== 执行路径 ===")
    typer.echo(output.route.value)
    typer.echo("\n=== 回答 ===")
    typer.echo(output.answer)

    trace_items = output.trace
    react_steps = output.data.get("react_steps", []) if isinstance(output.data, dict) else []

    if show_react_trace and isinstance(react_steps, list) and react_steps:
        # 这个代码块用于在结构化轨迹可用时隐藏冗长的原始ReAct行，避免重复展示。
        trace_items = [item for item in trace_items if not str(item).startswith("ReAct:")]

    if not show_react_trace:
        # 这个代码块用于默认隐藏ReAct细粒度轨迹，只展示主调度路径。
        trace_items = [item for item in output.trace if not str(item).startswith("ReAct:")]

    if trace_items:
        typer.echo("\n=== 调度轨迹 ===")
        for item in trace_items:
            typer.echo(f"- {item}")

    if show_react_trace and isinstance(react_steps, list) and react_steps:
        typer.echo("\n=== ReAct 详细轨迹 ===")
        _print_structured_react_steps(react_steps)

    if output.sources:
        typer.echo("\n=== 参考来源 ===")
        for source in output.sources:
            typer.echo(f"- {source}")


def _to_memory_scope(scope: str) -> MemoryScope:
    # 这个函数用于把命令行scope参数转换为枚举值。
    normalized = scope.strip().lower()
    mapping = {
        "session": MemoryScope.SESSION,
        "user": MemoryScope.USER,
        "project": MemoryScope.PROJECT,
    }
    if normalized not in mapping:
        raise typer.BadParameter("scope仅支持session/user/project。")
    return mapping[normalized]


@app.command("build-index")
def build_index_cmd(
    max_pdfs: int = typer.Option(0, "--max-pdfs", help="仅处理前N个PDF，0表示全部"),
    quiet: bool = typer.Option(False, "--quiet", help="减少过程日志输出"),
) -> None:
    settings = load_settings()
    typer.echo(
        f"启动 build-index: knowledge_dir={settings.knowledge_dir}, "
        f"enable_ocr={settings.enable_ocr}, ocr_dpi={settings.ocr_dpi}, max_pdfs={max_pdfs}"
    )
    logger = None if quiet else (lambda message: print(message, flush=True))
    pdf_count, chunk_count = build_vector_index(
        settings,
        max_pdfs=(None if max_pdfs <= 0 else max_pdfs),
        logger=logger,
    )
    typer.echo(f"索引构建完成: PDF={pdf_count}, 切片={chunk_count}")


@app.command("ask")
def ask_cmd(
    question: str = typer.Option(..., "--question", "-q", help="提问内容"),
    user_id: str = typer.Option("", "--user-id", help="用户身份ID（用于用户记忆）"),
    session_id: str = typer.Option("", "--session-id", help="会话ID（用于短期记忆）"),
    show_react_trace: bool = typer.Option(False, "--show-react-trace", help="是否显示ReAct轨迹"),
) -> None:
    # 这个命令用于执行单次全功能调度问答（Agent+ReAct）。
    settings = load_settings()
    if not has_index(settings):
        raise typer.BadParameter("未检测到向量索引，请先执行 build-index。")

    router = AgentRouter(settings)
    output = router.ask(question, user_id=user_id, session_id=session_id)
    _render_route_output(output, show_react_trace)


@app.command("chat")
def chat_cmd(
    user_id: str = typer.Option("", "--user-id", help="用户身份ID（用于用户记忆）"),
    session_id: str = typer.Option("", "--session-id", help="会话ID（默认自动生成）"),
    show_react_trace: bool = typer.Option(False, "--show-react-trace", help="是否显示ReAct轨迹"),
) -> None:
    # 这个命令用于执行交互式全功能调度问答（Agent+ReAct）。
    settings = load_settings()
    if not has_index(settings):
        raise typer.BadParameter("未检测到向量索引，请先执行 build-index。")

    router = AgentRouter(settings)
    active_session_id = session_id.strip() or f"chat-{uuid.uuid4()}"
    typer.echo("进入全功能交互模式，输入 exit 退出。")
    typer.echo(f"当前session_id: {active_session_id}")
    while True:
        question = typer.prompt("\n你")
        if question.strip().lower() in {"exit", "quit"}:
            typer.echo("已退出。")
            break

        output = router.ask(question, user_id=user_id, session_id=active_session_id)
        typer.echo("\n助手:")
        _render_route_output(output, show_react_trace)


@app.command("mcp-tools")
def mcp_tools_cmd() -> None:
    # 这个命令用于查看已注册的MCP工具名称。
    dispatcher = MCPDispatcher()
    typer.echo("可用MCP工具：")
    for tool in dispatcher.list_tools():
        typer.echo(f"- {tool}")


@app.command("mcp-call")
def mcp_call_cmd(payload: str = typer.Option(..., "--payload", "-p", help="MCP JSON请求体")) -> None:
    # 这个命令用于执行单次MCP JSON协议调用。
    dispatcher = MCPDispatcher()
    response = dispatcher.handle_json(payload)
    typer.echo(json.dumps(response, ensure_ascii=False, indent=2))


@app.command("mcp-demo")
def mcp_demo_cmd() -> None:
    # 这个命令用于快速演示阶段三的两个核心财务工具。
    dispatcher = MCPDispatcher()
    demo_requests = [
        {
            "tool_name": "calc_pe",
            "params": {
                "price": 8.25,
                "eps": 0.68,
            },
            "request_id": "demo-pe-001",
        },
        {
            "tool_name": "calc_debt_ratio",
            "params": {
                "total_asset": 217867,
                "total_liability": 199242,
            },
            "request_id": "demo-debt-001",
        },
    ]

    for request in demo_requests:
        typer.echo("\n=== MCP请求 ===")
        typer.echo(json.dumps(request, ensure_ascii=False, indent=2))
        typer.echo("=== MCP响应 ===")
        typer.echo(json.dumps(dispatcher.handle(request), ensure_ascii=False, indent=2))


@app.command("mcp-nl")
def mcp_nl_cmd(question: str = typer.Option(..., "--question", "-q", help="自然语言提问内容")) -> None:
    # 这个命令用于把自然语言问题自动转换成MCP请求并执行。
    settings = load_settings()
    dispatcher = MCPDispatcher()
    request = build_request_from_question(question, settings)
    response = dispatcher.handle(request)
    typer.echo("=== MCP请求 ===")
    typer.echo(json.dumps(request, ensure_ascii=False, indent=2))
    typer.echo("=== MCP响应 ===")
    typer.echo(json.dumps(response, ensure_ascii=False, indent=2))


@app.command("agent-route")
def agent_route_cmd(question: str = typer.Option(..., "--question", "-q", help="提问内容")) -> None:
    # 这个命令用于仅查看阶段四意图识别与分流结果。
    settings = load_settings()
    router = AgentRouter(settings)
    intent = router.route_only(question)
    typer.echo("=== 路由结果 ===")
    typer.echo(json.dumps({"route": intent.route.value, "reason": intent.reason}, ensure_ascii=False, indent=2))


@app.command("agent-ask")
def agent_ask_cmd(
    question: str = typer.Option(..., "--question", "-q", help="提问内容"),
    user_id: str = typer.Option("", "--user-id", help="用户身份ID（用于用户记忆）"),
    session_id: str = typer.Option("", "--session-id", help="会话ID（用于短期记忆）"),
    show_react_trace: bool = typer.Option(False, "--show-react-trace", help="是否显示ReAct轨迹"),
) -> None:
    # 这个命令用于执行阶段四完整调度链路。
    settings = load_settings()
    router = AgentRouter(settings)
    output = router.ask(question, user_id=user_id, session_id=session_id)
    _render_route_output(output, show_react_trace)


@app.command("memory-put")
def memory_put_cmd(
    scope: str = typer.Option(..., "--scope", help="记忆作用域: session/user/project"),
    text: str = typer.Option(..., "--text", help="记忆内容"),
    user_id: str = typer.Option("", "--user-id", help="用户ID"),
    session_id: str = typer.Option("", "--session-id", help="会话ID"),
    importance: float = typer.Option(0.5, "--importance", help="重要度[0,1]"),
) -> None:
    # 这个命令用于手动写入一条记忆记录。
    settings = load_settings()
    store = HybridMemoryStore(settings)
    memory_scope = _to_memory_scope(scope)
    expire_at = store.default_expiration()
    item_id = store.put(
        scope=memory_scope,
        text=text,
        user_id=user_id,
        session_id=session_id,
        tags=["manual_put"],
        payload={},
        importance=importance,
        expires_at=expire_at,
    )
    typer.echo(json.dumps({"ok": True, "item_id": item_id, "expires_at": expire_at.isoformat()}, ensure_ascii=False, indent=2))


@app.command("memory-search")
def memory_search_cmd(
    query: str = typer.Option(..., "--query", "-q", help="搜索查询"),
    scope: str = typer.Option(..., "--scope", help="记忆作用域: session/user/project"),
    user_id: str = typer.Option("", "--user-id", help="用户ID"),
    session_id: str = typer.Option("", "--session-id", help="会话ID"),
    top_k: int = typer.Option(5, "--top-k", help="返回条数"),
) -> None:
    # 这个命令用于按语义召回记忆记录。
    settings = load_settings()
    store = HybridMemoryStore(settings)
    memory_scope = _to_memory_scope(scope)
    rows = store.recall(
        query=query,
        scope=memory_scope,
        top_k=max(1, top_k),
        user_id=user_id,
        session_id=session_id,
    )
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "count": len(rows),
                "items": [
                    {
                        "item_id": row.item_id,
                        "scope": row.scope.value,
                        "user_id": row.user_id,
                        "session_id": row.session_id,
                        "text": row.text,
                        "tags": row.tags,
                        "importance": row.importance,
                        "updated_at": row.updated_at.isoformat(),
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("memory-list")
def memory_list_cmd(
    scope: str = typer.Option(..., "--scope", help="记忆作用域: session/user/project"),
    user_id: str = typer.Option("", "--user-id", help="用户ID"),
    session_id: str = typer.Option("", "--session-id", help="会话ID"),
    limit: int = typer.Option(10, "--limit", help="返回条数"),
) -> None:
    # 这个命令用于按时间列出最近记忆。
    settings = load_settings()
    store = HybridMemoryStore(settings)
    memory_scope = _to_memory_scope(scope)
    rows = store.list_recent(
        scope=memory_scope,
        limit=max(1, limit),
        user_id=user_id,
        session_id=session_id,
    )
    typer.echo(
        json.dumps(
            {
                "ok": True,
                "count": len(rows),
                "items": [
                    {
                        "item_id": row.item_id,
                        "scope": row.scope.value,
                        "user_id": row.user_id,
                        "session_id": row.session_id,
                        "text": row.text,
                        "tags": row.tags,
                        "importance": row.importance,
                        "updated_at": row.updated_at.isoformat(),
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("memory-prune")
def memory_prune_cmd() -> None:
    # 这个命令用于清理所有已过期记忆。
    settings = load_settings()
    store = HybridMemoryStore(settings)
    removed = store.prune_expired(now=datetime.now(store.default_expiration().tzinfo))
    typer.echo(json.dumps({"ok": True, "removed": removed}, ensure_ascii=False, indent=2))


@app.command("memory-clear")
def memory_clear_cmd(
    scope: str = typer.Option("", "--scope", help="可选，session/user/project"),
    user_id: str = typer.Option("", "--user-id", help="可选，用户ID"),
    session_id: str = typer.Option("", "--session-id", help="可选，会话ID"),
) -> None:
    # 这个命令用于按过滤条件清空记忆。
    settings = load_settings()
    store = HybridMemoryStore(settings)
    memory_scope = _to_memory_scope(scope) if scope.strip() else None
    removed = store.clear(scope=memory_scope, user_id=user_id, session_id=session_id)
    typer.echo(json.dumps({"ok": True, "removed": removed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
