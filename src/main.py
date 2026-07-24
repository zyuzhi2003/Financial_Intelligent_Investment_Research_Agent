# 这个文件用于提供命令行入口，支持构建索引、单次问答和交互式问答。
from __future__ import annotations

import json
import typer

from src.agent import AgentRouter
from src.config import load_settings
from src.mcp import MCPDispatcher
from src.mcp.nl_router import build_request_from_question
from src.rag.build_index import build_vector_index
from src.rag.qa_chain import ask_with_rag
from src.rag.retriever import has_index

app = typer.Typer(help="金融RAG+MCP智能问答系统")


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
def ask_cmd(question: str = typer.Option(..., "--question", "-q", help="提问内容")) -> None:
    settings = load_settings()
    if not has_index(settings):
        raise typer.BadParameter("未检测到向量索引，请先执行 build-index。")

    result = ask_with_rag(settings, question)
    typer.echo("\n=== 回答 ===")
    typer.echo(result["answer"])
    typer.echo("\n=== 参考来源 ===")
    for item in result["sources"]:
        typer.echo(f"- {item}")


@app.command("chat")
def chat_cmd() -> None:
    settings = load_settings()
    if not has_index(settings):
        raise typer.BadParameter("未检测到向量索引，请先执行 build-index。")

    typer.echo("进入交互模式，输入 exit 退出。")
    while True:
        question = typer.prompt("\n你")
        if question.strip().lower() in {"exit", "quit"}:
            typer.echo("已退出。")
            break

        result = ask_with_rag(settings, question)
        typer.echo("\n助手:")
        typer.echo(result["answer"])
        typer.echo("\n参考来源:")
        for item in result["sources"]:
            typer.echo(f"- {item}")


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
def agent_ask_cmd(question: str = typer.Option(..., "--question", "-q", help="提问内容")) -> None:
    # 这个命令用于执行阶段四完整调度链路。
    settings = load_settings()
    router = AgentRouter(settings)
    output = router.ask(question)
    typer.echo("=== 执行路径 ===")
    typer.echo(output.route.value)
    typer.echo("\n=== 回答 ===")
    typer.echo(output.answer)
    if output.trace:
        typer.echo("\n=== 调度轨迹 ===")
        for item in output.trace:
            typer.echo(f"- {item}")
    if output.sources:
        typer.echo("\n=== 参考来源 ===")
        for source in output.sources:
            typer.echo(f"- {source}")


if __name__ == "__main__":
    app()
