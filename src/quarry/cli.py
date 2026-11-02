"""Command line: ``quarry ask | chat | serve | mcp | eval | demo | data``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from quarry.agent import AgentEvent, QuarryAgent, open_agent
from quarry.config import QuarryConfig, load_config
from quarry.errors import QuarryError

app = typer.Typer(
    name="quarry",
    help="Ask your database anything: a LangGraph agent that plans, writes, verifies and "
    "self-corrects SQL, and shows its work.",
    no_args_is_help=True,
    add_completion=False,
)
demo_app = typer.Typer(help="Build the demo marketing warehouse.", no_args_is_help=True)
data_app = typer.Typer(help="Download benchmark data.", no_args_is_help=True)
eval_app = typer.Typer(help="Measure quarry (offline metrics now, LLM benchmark with a key).")
app.add_typer(demo_app, name="demo")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")

console = Console()
err = Console(stderr=True)

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="TOML config file.")]
DbOpt = Annotated[Path | None, typer.Option("--db", help="SQLite or DuckDB file to query.")]
OfflineOpt = Annotated[
    bool, typer.Option("--offline", help="Use the scripted demo model (no API key, no LLM).")
]
ThreadOpt = Annotated[
    str | None, typer.Option("--thread", help="Conversation id; reuse it for follow-ups.")
]


def _config(path: Path | None) -> QuarryConfig:
    # Search for .env from the working directory, not from this file's location: a bare
    # load_dotenv() walks up from src/quarry and would pick up the repo's .env from anywhere.
    load_dotenv(find_dotenv(usecwd=True))
    return load_config(path)


def _fail(exc: Exception) -> typer.Exit:
    err.print(f"[bold red]error:[/] {exc}")
    return typer.Exit(code=2)


# ---------------------------------------------------------------------------- ask / chat


def _render_event(event: AgentEvent, verbose: bool) -> None:
    data = event.data
    if event.kind == "link_schema":
        console.print(f"[dim]linked tables:[/] {', '.join(data.get('selected', []))}")
    elif event.kind in {"plan", "generate"}:
        for call in data.get("tool_calls", []):
            if call["name"] != "run_sql":
                console.print(f"[cyan]tool[/] {call['name']}({json.dumps(call['args'])})")
        if data.get("plan") and verbose:
            console.print(Panel(data["plan"], title="plan", border_style="cyan"))
        if "sql" in data:
            console.print(Syntax(data["sql"], "sql", theme="ansi_dark", word_wrap=True))
    elif event.kind == "check" and not data.get("ok"):
        console.print(f"[red]x[/] {data.get('feedback')}")
    elif event.kind == "execute":
        if data.get("ok"):
            console.print(f"[green]ok[/] {data['row_count']} rows in {data['elapsed_ms']} ms")
        else:
            console.print(f"[red]x[/] {data.get('error')}")
    elif event.kind == "verify" and not data.get("accepted"):
        console.print(f"[yellow]![/] {data.get('feedback')}")


def _render_outcome(outcome: dict[str, Any]) -> None:
    if outcome["columns"]:
        table = Table(show_lines=False, header_style="bold")
        for column in outcome["columns"]:
            table.add_column(str(column))
        for row in outcome["rows"][:20]:
            table.add_row(*["NULL" if v is None else str(v) for v in row])
        console.print(table)
        if outcome["row_count"] > 20:
            console.print(f"[dim]... {outcome['row_count']} rows in total[/]")
    style = "green" if outcome["status"] == "ok" else "red"
    console.print(
        Panel(outcome["answer"], title=f"answer ({outcome['status']})", border_style=style)
    )
    for warning in outcome["warnings"]:
        console.print(f"[yellow]warning:[/] {warning}")
    trace = outcome["trace"]
    estimated = " (estimated)" if trace.get("tokens_estimated") else ""
    console.print(
        f"[dim]{trace['llm_calls']} LLM calls, {trace['tool_calls']} tool calls, "
        f"{trace['tokens_in'] + trace['tokens_out']} tokens{estimated}, "
        f"thread {outcome['thread_id']}, trace {trace['path']}[/]"
    )


def _run(agent: QuarryAgent, question: str, thread: str | None, verbose: bool) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    for event in agent.stream(question, thread_id=thread):
        if event.kind == "done":
            outcome = event.data
        else:
            _render_event(event, verbose)
    return outcome


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="A question about the database.")],
    db: DbOpt = None,
    config: ConfigOpt = None,
    offline: OfflineOpt = False,
    thread: ThreadOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the outcome as JSON.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show the plan.")] = False,
) -> None:
    """Answer one question (add --thread to continue a conversation)."""
    try:
        cfg = _config(config)
        memory = cfg.memory_path if thread else None
        with open_agent(cfg, db_path=db, offline=offline, memory_path=memory) as agent:
            if as_json:
                print(json.dumps(agent.ask(question, thread_id=thread).to_dict(), default=str))
                return
            _render_outcome(_run(agent, question, thread, verbose))
    except (QuarryError, ValueError) as exc:
        raise _fail(exc) from exc


@app.command()
def chat(
    db: DbOpt = None,
    config: ConfigOpt = None,
    offline: OfflineOpt = False,
    thread: ThreadOpt = None,
) -> None:
    """Interactive conversation with memory (/new starts over, /quit leaves)."""
    from quarry.tracing import check_id, new_id

    try:
        cfg = _config(config)
        with open_agent(cfg, db_path=db, offline=offline, memory_path=cfg.memory_path) as agent:
            current = new_id() if thread is None else check_id(thread)
            console.print(
                f"[bold]quarry[/] on {agent.deps.database.path} ({agent.deps.database.dialect}), "
                f"model {'scripted demo' if offline else agent.deps.model_name}, thread {current}"
            )
            while True:
                try:
                    question = console.input("[bold cyan]> [/]").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question in {"/quit", "/exit"}:
                    break
                if question == "/new":
                    current = new_id()
                    console.print(f"[dim]new thread {current}[/]")
                    continue
                if question:
                    _render_outcome(_run(agent, question, current, verbose=False))
    except (QuarryError, ValueError) as exc:
        raise _fail(exc) from exc


# ---------------------------------------------------------------------------- servers


@app.command()
def serve(
    db: DbOpt = None,
    config: ConfigOpt = None,
    offline: OfflineOpt = False,
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Port.")] = None,
) -> None:
    """Start the web UI (streams every agent step)."""
    import uvicorn

    from quarry.web import create_app

    try:
        cfg = _config(config)
        with open_agent(cfg, db_path=db, offline=offline, memory_path=cfg.memory_path) as agent:
            agent.deps.database.catalog()
            web = create_app(agent, offline=offline)
            bind_host, bind_port = host or cfg.server.host, port or cfg.server.port
            err.print(f"quarry UI on http://{bind_host}:{bind_port}")
            uvicorn.run(web, host=bind_host, port=bind_port, log_level="warning")
    except QuarryError as exc:
        raise _fail(exc) from exc


@app.command()
def mcp(db: DbOpt = None, config: ConfigOpt = None, offline: OfflineOpt = False) -> None:
    """Serve quarry's tools and `ask` over MCP (stdio)."""
    from quarry.mcp_server import build_mcp_server

    try:
        cfg = _config(config)
        with open_agent(cfg, db_path=db, offline=offline, memory_path=cfg.memory_path) as agent:
            build_mcp_server(agent).run("stdio")
    except QuarryError as exc:
        raise _fail(exc) from exc


# ---------------------------------------------------------------------------- data


@demo_app.command("build")
def demo_build(
    out: Annotated[Path, typer.Option(help="Where to write the SQLite warehouse.")] = Path(
        "data/warehouse.sqlite"
    ),
    duckdb_copy: Annotated[
        bool, typer.Option("--duckdb", help="Also write a DuckDB copy next to it.")
    ] = False,
) -> None:
    """Download the Hillstrom e-mail experiment (4 MB, checksummed) and build the warehouse."""
    from quarry.datasets import HILLSTROM, fetch
    from quarry.demo import build_warehouse, export_duckdb

    try:
        csv_path = fetch(HILLSTROM, Path("data/raw"))
        summary = build_warehouse(csv_path, out)
        console.print(
            f"built {summary.path}: {summary.campaigns} campaigns, {summary.customers:,} "
            f"customers, {summary.sends:,} sends"
        )
        if duckdb_copy:
            duck = export_duckdb(out, out.with_suffix(".duckdb"))
            console.print(f"built {duck}")
    except QuarryError as exc:
        raise _fail(exc) from exc


@data_app.command("spider")
def data_spider(
    root: Annotated[Path, typer.Option(help="Data directory.")] = Path("data"),
    schemas_only: Annotated[
        bool,
        typer.Option(
            "--schemas-only",
            help="Only questions, gold SQL and schemas (2 MB, from the authors' mirrors). "
            "Enough for the guard and linking metrics; no database files.",
        ),
    ] = False,
) -> None:
    """Download Spider 1.0 (206 MB zip, checksummed) and unpack its dev split."""
    from quarry.eval.spider import prepare_spider, prepare_spider_metadata

    try:
        if schemas_only:
            layout = prepare_spider_metadata(root)
            console.print(f"Spider questions and schemas ready in {layout.root} (no databases)")
            return
        layout = prepare_spider(root)
        console.print(f"Spider dev ready in {layout.root} ({len(layout.dev_db_ids())} databases)")
    except QuarryError as exc:
        raise _fail(exc) from exc


# ---------------------------------------------------------------------------- eval


@eval_app.command("offline")
def eval_offline(
    root: Annotated[Path, typer.Option(help="Data directory.")] = Path("data"),
    out: Annotated[Path, typer.Option(help="Results directory.")] = Path("docs/results"),
) -> None:
    """Everything measurable without an LLM: linking recall, guard and red-team, latency."""
    from quarry.eval.offline import run_offline_suite

    try:
        summary = run_offline_suite(root, out, console=console)
        console.print(f"wrote {summary}")
    except QuarryError as exc:
        raise _fail(exc) from exc


@eval_app.command("benchmark")
def eval_benchmark(
    root: Annotated[Path, typer.Option(help="Data directory.")] = Path("data"),
    out: Annotated[Path, typer.Option(help="Results directory.")] = Path("docs/results"),
    config: ConfigOpt = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    run_name: Annotated[str | None, typer.Option(help="Name of the run directory.")] = None,
) -> None:
    """Execution accuracy on Spider dev with the configured OpenRouter model (needs a key)."""
    from quarry.eval.benchmark import run_benchmark

    try:
        cfg = _config(config)
        report = run_benchmark(
            cfg, root=root, out=out, limit=limit, run_name=run_name, console=console
        )
        console.print(f"wrote {report}")
    except QuarryError as exc:
        raise _fail(exc) from exc


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
