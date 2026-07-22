# 这个文件用于提供命令行入口，支持构建索引、单次问答和交互式问答。
from __future__ import annotations

import typer

from src.config import load_settings
from src.rag.build_index import build_vector_index
from src.rag.qa_chain import ask_with_rag
from src.rag.retriever import has_index

app = typer.Typer(help="金融纯RAG问答系统")


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


if __name__ == "__main__":
    app()
