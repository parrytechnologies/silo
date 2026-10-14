"""Model Context Protocol server: lets an MCP client (Claude Desktop, Claude Code, or any
other) search the local index and read passages as tools.

The server speaks MCP over stdio, opens the index store read-only, and makes no
network connections itself. What the *client* does with the passages it receives is
the client's business: a cloud-hosted assistant will send them to its provider.
"""

from __future__ import annotations

import threading
from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError

from silo import __version__
from silo.config import Settings
from silo.search import MODES

INSTRUCTIONS = (
    "SILO is a local search index over the user's own documents. Start with find_sources: "
    "it returns the files relevant to a question, each with its absolute path and the "
    "sections and pages that matched. If you can read files, open those files at the "
    "listed sections and answer from them: the file on disk is authoritative, and the "
    "index holds the text as it was when last indexed. A source whose state is 'modified' "
    "has changed since then; one that is 'missing' is not at that path on this machine. "
    "When you cannot read files, or the source is a long PDF or office document, use "
    "search_documents for the matching passages, get_passage_context to read around one, "
    "or read_document to read a document in order; these return the indexed text. Prefer "
    "several focused searches over one broad one, and try mode='keyword' for exact "
    "identifiers. Always cite the file and page you used, and say so when the documents "
    "do not cover the question."
)
MODE_HELP = (
    "'hybrid' (default: keyword + semantic, reranked), 'semantic' (meaning only) "
    "or 'keyword' (exact terms, identifiers, codes)."
)
MAX_TEXT = 4000  # characters per passage returned to the client
# Every tool only reads the local index: tell clients so they can skip confirmation prompts.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + " …[truncated]"


def _passage(row: dict, doc: Any | None) -> dict:
    out = {
        "passage_id": row["id"],
        "document_id": row["doc_id"],
        "position": row["chunk_index"],
        "file": doc["rel_path"] if doc else "",
        "path": doc["path"] if doc else "",
        "section": row["headings"],
        "text": _clip(row["text"]),
    }
    if row["page_start"]:
        out["page"] = row["page_start"]
        if row["page_end"] > row["page_start"]:
            out["page_end"] = row["page_end"]
    return out


def build_server(settings: Settings) -> FastMCP:
    server = FastMCP("silo", instructions=INSTRUCTIONS)
    lock = threading.Lock()
    state: dict[str, Any] = {}

    def engine():
        # Opened on first use so the client's handshake is instant; models load once.
        with lock:
            if "engine" not in state:
                from silo.engine import open_engine

                try:
                    state["engine"] = open_engine(settings, read_only=True)
                except RuntimeError as exc:  # no store yet, or models not downloaded
                    raise ToolError(str(exc)) from exc
            return state["engine"]

    @server.tool(annotations=READ_ONLY)
    def find_sources(
        query: Annotated[str, Field(description="A question in plain language, or exact terms.")],
        limit: Annotated[int, Field(ge=1, le=50, description="How many files.")] = 8,
        mode: Annotated[str, Field(description=MODE_HELP)] = "hybrid",
    ) -> dict:
        """Find the files relevant to a question. Each source is one file, with its
        absolute path, the sections and pages that matched, and whether the file has
        changed since it was indexed. Use this to decide which files to open."""
        if mode not in MODES:
            raise ToolError(f"mode must be one of: {', '.join(MODES)}")
        result = engine().searcher.sources(query, k=limit, mode=mode)
        return {
            "query": result.query,
            "mode": result.mode,
            "sources": [
                {
                    "path": s.path, "file": s.rel_path, "title": s.title,
                    "document_id": s.doc_id, "score": round(s.score, 4),
                    "reranked": s.rerank is not None, "matches": s.matches,
                    "state": s.source_state,
                    **({"also_at": s.also_at} if s.also_at else {}),
                    "sections": [
                        {"section": sec.headings, "passage_id": sec.passage_id,
                         **({"page": sec.page_start} if sec.page_start else {})}
                        for sec in s.sections
                    ],
                    "snippet": s.snippet,
                }
                for s in result.sources
            ],
        }

    @server.tool(annotations=READ_ONLY)
    def search_documents(
        query: Annotated[str, Field(description="A question in plain language, or exact terms.")],
        limit: Annotated[int, Field(ge=1, le=25, description="How many passages.")] = 8,
        mode: Annotated[str, Field(description=MODE_HELP)] = "hybrid",
    ) -> dict:
        """Search the user's indexed documents and return the best-matching passages,
        each with its file, page, section heading and relevance score."""
        if mode not in MODES:
            raise ToolError(f"mode must be one of: {', '.join(MODES)}")
        result = engine().searcher.search(query, k=limit, mode=mode)
        return {
            "query": result.query,
            "mode": result.mode,
            "results": [
                {
                    "passage_id": h.chunk_id, "document_id": h.doc_id, "file": h.rel_path,
                    "path": h.path, "state": h.source_state,
                    **({"also_at": h.also_at} if h.also_at else {}),
                    "title": h.title, **({"page": h.page_start} if h.page_start else {}),
                    "section": h.headings, "score": round(h.score, 4), "text": _clip(h.text),
                }
                for h in result.hits
            ],
        }

    @server.tool(annotations=READ_ONLY)
    def get_passage_context(
        passage_id: Annotated[str, Field(description="A passage_id from search_documents.")],
        before: Annotated[int, Field(ge=0, le=10)] = 2,
        after: Annotated[int, Field(ge=0, le=10)] = 2,
    ) -> dict:
        """Return a passage together with the passages just before and after it in the
        same document, in reading order. Use this to check a search result in context."""
        doc_id, _, index = passage_id.rpartition(":")
        if not doc_id or not index.isdigit():
            raise ToolError("passage_id should look like '<document_id>:<number>'")
        store = engine().store
        doc = store.document(doc_id)
        if doc is None:
            raise ToolError(f"no document with id {doc_id!r} in this index")
        rows = store.passages(doc_id, max(0, int(index) - before), int(index) + after)
        return {"file": doc["rel_path"], "path": doc["path"], "title": doc["title"],
                "passages": [_passage(r, doc) for r in rows]}

    @server.tool(annotations=READ_ONLY)
    def read_document(
        document_id: Annotated[str, Field(description="A document_id from search or list.")],
        start: Annotated[int, Field(ge=0, description="Position of the first passage.")] = 0,
        count: Annotated[int, Field(ge=1, le=20, description="How many passages.")] = 10,
    ) -> dict:
        """Read a document's passages in order, starting at a position. Page through a
        long document by calling again with start = next_start."""
        store = engine().store
        doc = store.document(document_id)
        if doc is None:
            raise ToolError(f"no document with id {document_id!r} in this index")
        rows = store.passages(document_id, start, start + count - 1)
        out: dict[str, Any] = {
            "file": doc["rel_path"], "path": doc["path"], "title": doc["title"],
            "pages": doc["pages"],
            "total_passages": doc["chunks"], "passages": [_passage(r, doc) for r in rows],
        }
        if start + count < doc["chunks"]:
            out["next_start"] = start + count
        return out

    @server.tool(annotations=READ_ONLY)
    def list_documents(
        contains: Annotated[str, Field(
            description="Only documents whose file name or title contains this text.")] = "",
        limit: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> dict:
        """List the documents in the index with their ids, titles and sizes."""
        needle = contains.lower().strip()
        docs = [
            d for d in engine().store.documents()
            if not needle or needle in d["rel_path"].lower() or needle in d["title"].lower()
        ]
        return {
            "total": len(docs),
            "documents": [
                {"document_id": d["doc_id"], "file": d["rel_path"], "path": d["path"],
                 "title": d["title"],
                 "pages": d["pages"], "passages": d["chunks"], "status": d["status"]}
                for d in docs[:limit]
            ],
        }

    @server.tool(annotations=READ_ONLY)
    def index_status() -> dict:
        """Describe the index: where it is, how much it holds and which models built it."""
        eng = engine()
        stats = eng.store.stats()
        manifest = eng.store.manifest()
        return {
            "silo_version": __version__,
            "index_store": str(settings.index_dir),
            "store_id": manifest.get("store_id"),
            "documents": stats["docs"], "pages": stats["pages"], "passages": stats["chunks"],
            "failed_documents": stats["failed"],
            "embedding_model": settings.embedding_model,
            "source_folders": eng.store.roots(),
        }

    return server


def run_stdio(settings: Settings) -> None:
    build_server(settings).run("stdio")
