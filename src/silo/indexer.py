"""Incremental indexing: walk a folder, parse what changed, drop what disappeared."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from silo.config import Settings
from silo.embed import Embedder
from silo.parse import Parser, supported_suffixes
from silo.store import Store, file_sha256

SKIP_DIRS = {".git", ".svn", "node_modules", "__pycache__", ".venv", ".silo"}


@dataclass
class IndexEvent:
    kind: str  # "indexed" | "unchanged" | "failed" | "empty" | "removed"
    path: str
    chunks: int = 0
    pages: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class IndexReport:
    root: str
    total: int = 0
    indexed: int = 0
    unchanged: int = 0
    failed: int = 0
    empty: int = 0
    removed: int = 0
    chunks: int = 0
    pages: int = 0
    seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def discover(root: Path, settings: Settings) -> Iterator[Path]:
    suffixes = supported_suffixes(settings)
    if root.is_file():
        if root.suffix.lower() in suffixes:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.startswith("~$"):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() in suffixes:
                yield path


class Indexer:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.parser = Parser(settings, embedder.tokenizer)

    def index(
        self,
        root: Path,
        force: bool = False,
        on_event: Callable[[IndexEvent, int, int], None] | None = None,
    ) -> IndexReport:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        base = root if root.is_dir() else root.parent
        started = time.perf_counter()
        self.store.check_embedding(self.settings.embedding_model, self.embedder.dim)

        files = list(discover(root, self.settings))
        report = IndexReport(root=str(base), total=len(files))
        seen: set[str] = set()

        for position, path in enumerate(files, start=1):
            event = self._index_one(path, base, force)
            seen.add(str(path))
            setattr(report, event.kind, getattr(report, event.kind) + 1)
            report.chunks += event.chunks
            report.pages += event.pages
            if event.kind == "failed":
                report.failures.append((event.path, event.error))
            if on_event:
                on_event(event, position, len(files))

        # Anything catalogued under this root that is no longer on disk leaves the index.
        if root.is_dir():
            for row in self.store.documents():
                if row["root"] == str(base) and row["path"] not in seen:
                    self.store.delete_document(row["doc_id"])
                    report.removed += 1
                    if on_event:
                        on_event(IndexEvent("removed", row["path"]), len(files), len(files))

        if report.indexed or report.removed or report.empty or report.failed:
            self.store.finalize()
        report.seconds = time.perf_counter() - started
        return report

    def _index_one(self, path: Path, base: Path, force: bool) -> IndexEvent:
        stat = path.stat()
        existing = self.store.document_by_path(str(path))
        # Keep a document's id stable even if the store was relocated from another path.
        doc_id = existing["doc_id"] if existing is not None else _doc_id(path)
        done = existing is not None and existing["status"] in ("indexed", "empty")

        if done and not force and existing["size"] == stat.st_size:
            if existing["mtime"] == stat.st_mtime:
                return IndexEvent("unchanged", str(path))
            if existing["sha256"] == file_sha256(path):  # touched, not edited
                self.store.touch_document(doc_id, stat.st_mtime)
                return IndexEvent("unchanged", str(path))

        record = dict(
            doc_id=doc_id, path=str(path), root=str(base),
            rel_path=str(path.relative_to(base)), suffix=path.suffix.lower(),
            size=stat.st_size, mtime=stat.st_mtime, sha256=file_sha256(path),
            indexed_at=time.time(),
        )
        started = time.perf_counter()
        try:
            parsed = self.parser.parse(path)
            vectors = self.embedder.embed_passages([c.search_text for c in parsed.chunks])
        except Exception as exc:  # one bad file must not stop the run
            self.store.delete_chunks(doc_id)
            error = f"{type(exc).__name__}: {exc}"[:500]
            self.store.upsert_document(
                **record, title=path.stem, pages=0, chunks=0, status="failed", error=error,
                parse_seconds=time.perf_counter() - started,
            )
            return IndexEvent("failed", str(path), error=error,
                              seconds=time.perf_counter() - started)

        self.store.delete_chunks(doc_id)
        self.store.add_chunks(doc_id, parsed.chunks, vectors)
        seconds = time.perf_counter() - started
        status = "indexed" if parsed.chunks else "empty"
        self.store.upsert_document(
            **record, title=parsed.title or path.stem, pages=parsed.pages,
            chunks=len(parsed.chunks), status=status, error="", parse_seconds=seconds,
        )
        return IndexEvent(status, str(path), chunks=len(parsed.chunks), pages=parsed.pages,
                          seconds=seconds)
