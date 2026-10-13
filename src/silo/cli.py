"""Command line interface."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from silo import __version__
from silo.config import Settings, load_settings

app = typer.Typer(
    help="SILO — Search and Indexing, Local-Only. Document search on your own CPU.",
    no_args_is_help=True, add_completion=False, rich_markup_mode=None,
)
models_app = typer.Typer(help="Download and inspect the local models.", no_args_is_help=True)
store_app = typer.Typer(help="Inspect, select and relocate index stores.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(store_app, name="store")
console = Console()
err = Console(stderr=True)

HomeOpt = Annotated[Path | None, typer.Option(
    "--home", help="Models and default index directory. Default: $SILO_HOME or ~/.silo")]
IndexOpt = Annotated[Path | None, typer.Option(
    "--index", help="Index store to use. Default: $SILO_INDEX, config.toml, or <home>/index")]


def _settings(home: Path | None, index: Path | None = None) -> Settings:
    settings = load_settings(home)
    if index is not None:
        settings.index = str(index)
    return settings


def _open(settings: Settings, read_only: bool | None = None):
    """Open the index store and its models, or exit with a readable message."""
    from silo import models
    from silo.engine import open_engine
    from silo.store import IndexMismatch, StoreNotFound

    try:
        engine = open_engine(settings, read_only=read_only)
    except (models.ModelsMissing, StoreNotFound, IndexMismatch) as exc:
        err.print(f"[red]{exc}[/red]", highlight=False)
        raise typer.Exit(2) from exc
    for notice in engine.notices:
        err.print(f"[dim]{notice}[/dim]", highlight=False)
    return engine


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"silo {__version__}")


# ---- models ----------------------------------------------------------------

@models_app.command("pull")
def models_pull(home: HomeOpt = None) -> None:
    """Download parsing, embedding and reranking models. The only command that uses the network."""
    from silo import models

    settings = _settings(home)
    code = models.pull_in_subprocess(settings)
    if code != 0:
        raise typer.Exit(code)
    _print_models(settings)


@models_app.command("status")
def models_status(home: HomeOpt = None) -> None:
    """Show which models are present on disk."""
    _print_models(_settings(home))


def _print_models(settings: Settings) -> None:
    from silo import models

    table = Table(title=f"Models in {settings.models_dir}")
    for col in ("role", "model", "present", "size"):
        table.add_column(col)
    for m in models.status(settings):
        table.add_row(m.role, m.name, "[green]yes[/green]" if m.present else "[red]no[/red]",
                      f"{m.size_mb:.0f} MB" if m.present else "—")
    console.print(table)


# ---- index stores ----------------------------------------------------------

@store_app.command("info")
def store_info(home: HomeOpt = None, index: IndexOpt = None) -> None:
    """Describe the index store in use: location, contents, models, source folders."""
    from silo.store import Store, StoreNotFound

    settings = _settings(home, index)
    try:
        store = Store(settings, read_only=True)
    except StoreNotFound as exc:
        err.print(f"[yellow]{exc}[/yellow]", highlight=False)
        raise typer.Exit(2) from exc
    manifest, stats = store.manifest(), store.stats()
    created = manifest.get("created_at")
    console.print(f"[bold]Index store[/bold] {settings.index_dir}", highlight=False)
    rows = [
        ("store id", manifest.get("store_id", "—")),
        ("created", time.strftime("%Y-%m-%d %H:%M", time.localtime(int(created)))
         if created else "—"),
        ("created with", manifest.get("created_with", "—")),
        ("embedding model", manifest.get("embedding_model", "— (empty store)")),
        ("passage size", f"{manifest.get('chunk_tokens', '—')} tokens"),
        ("documents", f"{stats['docs']} ({stats['indexed']} indexed, {stats['empty']} empty, "
                      f"{stats['failed']} failed)"),
        ("pages / passages", f"{stats['pages']} / {stats['chunks']}"),
        ("size on disk", f"{store.size_bytes() / 1_048_576:.1f} MB"),
        ("writable here", "yes" if os.access(settings.index_dir, os.W_OK) else "no"),
    ]
    for label, value in rows:
        console.print(f"  {label:<16} {value}", highlight=False)
    for root in store.roots():
        present = Path(root).exists()
        mark = "[green]present[/green]" if present else "[yellow]not on this machine[/yellow]"
        console.print(f"  source folder    {root}  ({mark})", highlight=False)


@store_app.command("use")
def store_use(
    path: Annotated[Path | None, typer.Argument(help="Index store directory.")] = None,
    read_only: Annotated[bool, typer.Option(
        "--read-only", help="Never write to this store (shared or borrowed index).")] = False,
    default: Annotated[bool, typer.Option(
        "--default", help="Go back to the built-in store at <home>/index.")] = False,
    home: HomeOpt = None,
) -> None:
    """Make an index store the default for every command (saved in config.toml)."""
    settings = _settings(home)
    if default == (path is not None):
        err.print("[red]Give a store path, or --default.[/red]")
        raise typer.Exit(2)
    if default:
        _write_config(settings.home, {"index": None, "read_only": None})
        console.print(f"Using the default index store at {settings.home / 'index'}",
                      highlight=False)
        return
    target = path.expanduser().resolve()
    _write_config(settings.home, {"index": str(target), "read_only": read_only})
    exists = (target / "catalogue.sqlite").is_file()
    console.print(
        f"Now using index store {target}" + (" (read-only)" if read_only else "")
        + ("" if exists else " — empty; `silo index <folder>` will create it"),
        highlight=False,
    )


def _write_config(home: Path, updates: dict[str, object]) -> None:
    """Set or remove top-level keys in config.toml, leaving everything else as written."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.toml"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines
             if not any(re.match(rf"\s*{re.escape(key)}\s*=", line) for key in updates)]
    for key, value in updates.items():
        if value is None:
            continue
        rendered = str(value).lower() if isinstance(value, bool) else json.dumps(str(value))
        lines.append(f"{key} = {rendered}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


@store_app.command("relocate")
def store_relocate(
    old: Annotated[str, typer.Argument(help="Source folder as recorded in the store.")],
    new: Annotated[Path, typer.Argument(help="Where those documents are on this machine.")],
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Tell a store its documents now live somewhere else. No re-indexing needed."""
    from silo.store import Store

    settings = _settings(home, index)
    store = Store(settings, read_only=False)
    moved = store.relocate(old.rstrip("/"), str(new.expanduser().resolve()))
    if not moved:
        err.print(f"[yellow]No documents are recorded under {old}.[/yellow] Recorded source "
                  f"folders: {', '.join(store.roots()) or 'none'}", highlight=False)
        raise typer.Exit(1)
    console.print(f"Re-pointed {moved} documents to {new.expanduser().resolve()}",
                  highlight=False)


# ---- indexing and search ---------------------------------------------------

@app.command()
def index(
    folder: Annotated[Path, typer.Argument(help="Folder (or single file) to index.")],
    force: Annotated[bool, typer.Option("--force", help="Re-parse unchanged files.")] = False,
    ocr: Annotated[bool | None, typer.Option(
        "--ocr/--no-ocr", help="OCR scanned pages and images (slower).")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Summary only (for cron).")] = False,
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Index a folder. Safe to re-run: only new and changed files are parsed."""
    settings = _settings(home, index)
    if ocr is not None:
        settings.ocr = ocr
    if settings.read_only:
        err.print(f"[red]The index store at {settings.index_dir} is configured read-only.[/red]",
                  highlight=False)
        raise typer.Exit(2)
    engine = _open(settings, read_only=False)
    from silo.indexer import Indexer
    from silo.store import IndexMismatch

    store = engine.store
    indexer = Indexer(settings, store, engine.embedder)
    marks = {"indexed": "[green]indexed  [/green]", "unchanged": "[dim]unchanged[/dim]",
             "failed": "[red]failed   [/red]", "empty": "[yellow]empty    [/yellow]",
             "removed": "[yellow]removed  [/yellow]"}

    def on_event(event, position, total):
        if quiet or event.kind == "unchanged":
            return
        detail = ""
        if event.kind == "indexed":
            detail = f"{event.pages or '—'} pages · {event.chunks} chunks · {event.seconds:.1f}s"
        elif event.kind == "failed":
            detail = event.error
        elif event.kind == "empty":
            detail = "no extractable text (scanned? try --ocr)"
        console.print(f"[dim]{position:>4}/{total}[/dim] {marks[event.kind]} "
                      f"{Path(event.path).name}  [dim]{detail}[/dim]", highlight=False)

    console.print(f"Indexing [bold]{folder}[/bold] into {settings.index_dir} on CPU "
                  f"({settings.num_threads} threads, OCR {'on' if settings.ocr else 'off'})",
                  highlight=False)
    try:
        report = indexer.index(folder, force=force, on_event=on_event)
    except (FileNotFoundError, IndexMismatch) as exc:
        err.print(f"[red]{exc}[/red]", highlight=False)
        raise typer.Exit(2) from exc
    rate = f" · {report.pages / report.seconds:.1f} pages/s" if report.pages else ""
    console.print(
        f"[bold]Done[/bold] in {report.seconds:.1f}s{rate} — {report.indexed} indexed, "
        f"{report.unchanged} unchanged, {report.empty} empty, {report.failed} failed, "
        f"{report.removed} removed. Index now holds {store.chunk_count()} passages.",
        highlight=False,
    )
    if report.failed:
        raise typer.Exit(1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    k: Annotated[int, typer.Option("-k", help="Number of results.")] = 5,
    mode: Annotated[str, typer.Option(help="hybrid | semantic | keyword")] = "hybrid",
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    passages: Annotated[bool, typer.Option(
        "--passages", help="List matching passages instead of the files they are in.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Search the index from the terminal: the relevant files and where they match."""
    engine = _open(_settings(home, index), read_only=True)
    if passages:
        result = engine.searcher.search(query, k=k, mode=mode, rerank=rerank)
        found = result.hits
    else:
        result = engine.searcher.sources(query, k=k, mode=mode, rerank=rerank)
        found = result.sources
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    if not found:
        console.print("[yellow]No results.[/yellow]")
        return
    for item in found:
        _print_hit(item) if passages else _print_source(item)
    timing = " · ".join(f"{name} {ms:.0f}ms" for name, ms in result.timings_ms.items())
    console.print(f"[dim]{timing}[/dim]", highlight=False)


def _print_source(source) -> None:
    state = {"modified": "  [yellow]changed since indexing[/yellow]",
             "missing": "  [red]file not found[/red]"}.get(source.source_state, "")
    matches = f"{source.matches} match{'es' if source.matches != 1 else ''}"
    console.print(f"[bold cyan]{source.rank:>2}[/bold cyan] [bold]{source.rel_path}[/bold]  "
                  f"[dim]{source.score:.2f} · {matches}[/dim]{state}", highlight=False)
    console.print(f"   {source.path}", highlight=False, markup=False, style="dim", soft_wrap=True)
    for other in source.also_at:
        console.print(f"   same file: {other}", highlight=False, markup=False, style="dim",
                      soft_wrap=True)
    for section in source.sections:
        page = f"p.{section.page_start}" if section.page_start else ""
        label = " · ".join(part for part in (section.headings, page) if part)
        if label:
            console.print(f"   § {label}", highlight=False, markup=False, style="blue")
    console.print(f"   {source.snippet}\n", highlight=False, markup=False)


def _print_hit(hit) -> None:
    page = f" · p.{hit.page_start}" if hit.page_start else ""
    ranks = " ".join(
        part for part in (
            f"vec#{hit.vector_rank}" if hit.vector_rank else "",
            f"bm25#{hit.keyword_rank}" if hit.keyword_rank else "",
            f"rerank {hit.rerank:.2f}" if hit.rerank is not None else "",
        ) if part
    )
    console.print(f"[bold cyan]{hit.rank:>2}[/bold cyan] [bold]{hit.rel_path}[/bold]{page}  "
                  f"[dim]{ranks}[/dim]", highlight=False)
    if hit.headings:
        console.print(f"   [blue]{hit.headings}[/blue]", highlight=False)
    snippet = " ".join(hit.text.split())
    console.print(f"   {snippet[:320]}{'…' if len(snippet) > 320 else ''}\n", highlight=False,
                  markup=False)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer from the indexed documents.")],
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Answer a question with the configured LLM, citing retrieved passages."""
    settings = _settings(home, index)
    from silo.llm import LLMError, egress, stream_answer
    from silo.search import reading_order

    if not settings.llm_enabled:
        err.print("[red]No LLM configured.[/red] Set SILO_LLM_BASE_URL and "
                  "SILO_LLM_MODEL (see docs/llm.md).")
        raise typer.Exit(2)
    engine = _open(settings, read_only=True)
    result = reading_order(engine.searcher.search(question, k=settings.llm_context_chunks))
    if not result.hits:
        console.print("[yellow]Nothing in the index matches that question.[/yellow]")
        return
    console.print(f"[dim]{settings.llm_model} · egress: {egress(settings)}[/dim]\n")
    try:
        for token in stream_answer(settings, question, result.hits):
            console.print(token, end="", highlight=False, markup=False, soft_wrap=True)
    except LLMError as exc:
        err.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print("\n\n[bold]Sources[/bold]")
    for n, hit in enumerate(result.hits, start=1):
        page = f", p.{hit.page_start}" if hit.page_start else ""
        console.print(f" [{n}] {hit.rel_path}{page}", highlight=False, markup=False)


# ---- serving ---------------------------------------------------------------

@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address. Default 127.0.0.1.")] = None,
    port: Annotated[int | None, typer.Option(help="Port. Default 8321.")] = None,
    read_only: Annotated[bool, typer.Option(
        "--read-only", help="Serve the index without ever writing to it.")] = False,
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Start the web interface."""
    import uvicorn

    from silo import models
    from silo.server import create_app
    from silo.store import StoreNotFound

    settings = _settings(home, index)
    settings.host = host or settings.host
    settings.port = port or settings.port
    settings.read_only = settings.read_only or read_only
    try:
        application = create_app(settings)
    except (models.ModelsMissing, StoreNotFound) as exc:
        err.print(f"[red]{exc}[/red]", highlight=False)
        raise typer.Exit(2) from exc
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        err.print("[yellow]Warning:[/yellow] binding beyond localhost. The engine has no "
                  "authentication; put it behind a reverse proxy that does.")
    console.print(f"SILO is at [bold]http://{settings.host}:{settings.port}[/bold] · index "
                  f"{settings.index_dir}" + (" (read-only)" if settings.read_only else ""),
                  highlight=False)
    uvicorn.run(application, host=settings.host, port=settings.port, log_level="warning")


@app.command()
def mcp(home: HomeOpt = None, index: IndexOpt = None) -> None:
    """Run a Model Context Protocol server on stdio, for Claude and other MCP clients."""
    settings = _settings(home, index)
    # MCP owns stdout: one stray byte corrupts the protocol stream. Keep the real stdout
    # for the protocol and send anything else written to file descriptor 1 (native
    # libraries included) to stderr.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")
    from silo.mcp_server import run_stdio

    run_stdio(settings)


# ---- housekeeping ----------------------------------------------------------

@app.command()
def status(home: HomeOpt = None, index: IndexOpt = None) -> None:
    """Show what is in the index."""
    from silo.store import Store, StoreNotFound

    settings = _settings(home, index)
    try:
        store = Store(settings, read_only=True)
    except StoreNotFound:
        console.print(f"[bold]Index[/bold] {settings.index_dir}\n  empty — run "
                      "`silo index <folder>`", highlight=False)
        return
    stats = store.stats()
    console.print(f"[bold]Index[/bold] {settings.index_dir}", highlight=False)
    console.print(f"  documents {stats['docs']}  (indexed {stats['indexed']}, empty "
                  f"{stats['empty']}, failed {stats['failed']})", highlight=False)
    console.print(f"  pages {stats['pages']}  passages {stats['chunks']}  "
                  f"source size {stats['bytes'] / 1_048_576:.1f} MB", highlight=False)
    for root in store.roots():
        console.print(f"  root: {root}", highlight=False)
    for row in store.documents():
        if row["status"] == "failed":
            console.print(f"  [red]failed[/red] {row['rel_path']}: {row['error']}",
                          highlight=False)


@app.command()
def doctor(home: HomeOpt = None, index: IndexOpt = None) -> None:
    """Check that this machine is ready to run offline."""
    from silo import models
    from silo.llm import egress

    settings = _settings(home, index)
    ok = True

    def line(passed: bool, text: str) -> None:
        console.print(f" {'[green]ok  [/green]' if passed else '[red]FAIL[/red]'}  {text}",
                      highlight=False)

    console.print(f"[bold]SILO {__version__}[/bold] · home {settings.home}", highlight=False)
    for m in models.status(settings):
        ok &= m.present
        line(m.present, f"{m.role} model on disk: {m.name}")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    ok &= offline
    line(offline, "model libraries forced offline (HF_HUB_OFFLINE=1)")
    line(True, f"inference device: cpu · {settings.num_threads} threads")
    has_store = settings.catalogue_path.is_file()
    line(True, f"index store: {settings.index_dir}"
               + ("" if has_store else " (empty)")
               + (" · read-only" if settings.read_only else ""))
    where = egress(settings)
    if where == "none":
        line(True, "LLM: not configured — no component makes outbound requests")
    elif where in ("loopback", "private"):
        line(True, f"LLM: {settings.llm_model} at {settings.llm_base_url} ({where} address)")
    else:
        console.print(f" [yellow]note[/yellow]  LLM: {settings.llm_model} at "
                      f"{settings.llm_base_url} is EXTERNAL — retrieved passages will leave "
                      "this machine when you use Ask.", highlight=False)
    if not ok:
        raise typer.Exit(1)


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    home: HomeOpt = None, index: IndexOpt = None,
) -> None:
    """Delete the index store in use (not the models, and never your documents)."""
    from silo.store import Store

    settings = _settings(home, index)
    if settings.read_only:
        err.print("[red]This index store is configured read-only.[/red]")
        raise typer.Exit(2)
    if not yes:
        typer.confirm(f"Delete the index at {settings.index_dir}?", abort=True)
    Store(settings, read_only=False).reset()
    console.print("Index deleted.")


if __name__ == "__main__":
    app()
