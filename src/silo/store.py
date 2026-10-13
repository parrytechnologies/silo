"""Storage: a SQLite catalogue of documents and a LanceDB table of chunks.

LanceDB is an embedded, file-based store: it holds the vectors and a native BM25
full-text index side by side, so there is no database server to run or secure.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pyarrow as pa

from silo.config import Settings

CHUNK_TABLE = "chunks"
# Below this many chunks an exact (brute-force) vector scan is both faster to build
# and perfectly accurate; above it we build an approximate index.
ANN_THRESHOLD = 100_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    root          TEXT NOT NULL,
    rel_path      TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    suffix        TEXT NOT NULL DEFAULT '',
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    sha256        TEXT NOT NULL,
    pages         INTEGER NOT NULL DEFAULT 0,
    chunks        INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    error         TEXT NOT NULL DEFAULT '',
    parse_seconds REAL NOT NULL DEFAULT 0,
    indexed_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexMismatch(RuntimeError):
    pass


class StoreNotFound(RuntimeError):
    pass


class ReadOnlyStore(RuntimeError):
    pass


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class Store:
    def __init__(self, settings: Settings, read_only: bool | None = None):
        self.settings = settings
        self.read_only = settings.read_only if read_only is None else read_only
        self._lock = threading.RLock()
        self._lance = None
        self._table = None
        if self.read_only:
            # A borrowed or shared store: open it without creating or changing anything.
            if not settings.catalogue_path.is_file():
                raise StoreNotFound(
                    f"No index store at {settings.index_dir}. Build one with "
                    "`silo index <folder>`, or point --index / SILO_INDEX at an existing store."
                )
            uri = f"file:{quote(str(settings.catalogue_path))}?mode=ro"
            self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            return
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(settings.catalogue_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def _writable(self) -> None:
        if self.read_only:
            raise ReadOnlyStore(
                f"The index store at {self.settings.index_dir} is open read-only."
            )

    # ---- catalogue -------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._writable()
        with self._lock:
            self._db.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._db.commit()

    def check_embedding(self, model: str, dim: int) -> None:
        """Vectors from different models are not comparable; refuse to mix them."""
        stored = self.get_meta("embedding_model")
        if stored is None:
            # First write: stamp the store so it can be identified and reopened elsewhere.
            from silo import __version__

            self.set_meta("embedding_model", model)
            self.set_meta("embedding_dim", str(dim))
            self.set_meta("chunk_tokens", str(self.settings.chunk_tokens))
            self.set_meta("created_at", str(int(time.time())))
            self.set_meta("created_with", f"silo {__version__}")
        elif stored != model:
            raise IndexMismatch(
                f"This index was built with embedding model {stored!r} but the current "
                f"setting is {model!r}. Run `silo reset` and re-index, or restore the setting."
            )
        if self.get_meta("store_id") is None:  # also stamps stores made before ids existed
            self.set_meta("store_id", uuid.uuid4().hex[:12])
        if self.get_meta("chunk_tokens") is None:
            self.set_meta("chunk_tokens", str(self.settings.chunk_tokens))

    def manifest(self) -> dict[str, str]:
        """Everything the store records about itself."""
        with self._lock:
            rows = self._db.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def relocate(self, old_root: str, new_root: str) -> int:
        """Re-point documents from one source folder to another (after a move or a copy
        to a different machine). Passages and vectors are untouched."""
        self._writable()
        old, new = Path(old_root), Path(new_root)
        moved = 0
        with self._lock:
            for row in self._db.execute("SELECT doc_id, path, root FROM documents").fetchall():
                path = Path(row["path"])
                try:
                    relative = path.relative_to(old)
                except ValueError:
                    continue
                root = Path(row["root"])
                new_doc_root = new / root.relative_to(old) if root.is_relative_to(old) else new
                self._db.execute(
                    "UPDATE documents SET path = ?, root = ? WHERE doc_id = ?",
                    (str(new / relative), str(new_doc_root), row["doc_id"]),
                )
                moved += 1
            self._db.commit()
        return moved

    def document_by_path(self, path: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()

    def document(self, doc_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()

    def documents(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute("SELECT * FROM documents ORDER BY rel_path").fetchall()

    def documents_by_ids(self, ids: list[str]) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({marks})", ids
            ).fetchall()
        return {r["doc_id"]: r for r in rows}

    def documents_by_sha(self, shas: list[str]) -> dict[str, list[sqlite3.Row]]:
        """Every catalogued file with one of these content hashes, grouped by hash."""
        if not shas:
            return {}
        marks = ",".join("?" for _ in shas)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM documents WHERE sha256 IN ({marks}) ORDER BY path", shas
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["sha256"], []).append(row)
        return grouped

    def roots(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT root FROM documents ORDER BY root").fetchall()
        return [r["root"] for r in rows]

    def upsert_document(self, **doc: Any) -> None:
        self._writable()
        doc.setdefault("indexed_at", time.time())
        cols = list(doc)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "doc_id")
        with self._lock:
            self._db.execute(
                f"INSERT INTO documents({', '.join(cols)}) VALUES({', '.join('?' for _ in cols)}) "
                f"ON CONFLICT(doc_id) DO UPDATE SET {updates}",
                [doc[c] for c in cols],
            )
            self._db.commit()

    def touch_document(self, doc_id: str, mtime: float) -> None:
        self._writable()
        with self._lock:
            self._db.execute("UPDATE documents SET mtime = ? WHERE doc_id = ?", (mtime, doc_id))
            self._db.commit()

    def delete_document(self, doc_id: str) -> None:
        self._writable()
        self.delete_chunks(doc_id)
        with self._lock:
            self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._db.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS docs, "
                "COALESCE(SUM(status = 'indexed'), 0) AS indexed, "
                "COALESCE(SUM(status = 'failed'), 0) AS failed, "
                "COALESCE(SUM(status = 'empty'), 0) AS empty, "
                "COALESCE(SUM(pages), 0) AS pages, COALESCE(SUM(chunks), 0) AS chunks, "
                "COALESCE(SUM(size), 0) AS bytes, MAX(indexed_at) AS last_indexed FROM documents"
            ).fetchone()
        return dict(row)

    # ---- chunks ----------------------------------------------------------

    def _open_table(self, dim: int | None = None, create: bool = False):
        if self._table is not None:
            return self._table
        import lancedb

        if self._lance is None:
            self._lance = lancedb.connect(str(self.settings.lance_dir))
        try:
            self._table = self._lance.open_table(CHUNK_TABLE)
        except Exception:
            if not create or dim is None:
                return None
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("text", pa.string()),
                    pa.field("search_text", pa.string()),
                    pa.field("headings", pa.string()),
                    pa.field("page_start", pa.int32()),
                    pa.field("page_end", pa.int32()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            )
            self._table = self._lance.create_table(CHUNK_TABLE, schema=schema)
        return self._table

    def add_chunks(self, doc_id: str, chunks: list, vectors: np.ndarray) -> None:
        self._writable()
        if not chunks:
            return
        table = self._open_table(dim=int(vectors.shape[1]), create=True)
        rows = [
            {
                "id": f"{doc_id}:{c.index}",
                "doc_id": doc_id,
                "chunk_index": c.index,
                "text": c.text,
                "search_text": c.search_text,
                "headings": " › ".join(c.headings),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "vector": vectors[i].tolist(),
            }
            for i, c in enumerate(chunks)
        ]
        with self._lock:
            table.add(rows)

    def delete_chunks(self, doc_id: str) -> None:
        self._writable()
        table = self._open_table()
        if table is not None:
            with self._lock:
                table.delete(f"doc_id = {_sql_quote(doc_id)}")

    def chunk_count(self) -> int:
        table = self._open_table()
        return table.count_rows() if table is not None else 0

    def finalize(self) -> None:
        """Rebuild the keyword index (and the ANN index for large corpora), then compact."""
        self._writable()
        table = self._open_table()
        if table is None or table.count_rows() == 0:
            return
        with self._lock, warnings.catch_warnings():
            # Native (non-tantivy) BM25 index. The call is stable across the LanceDB
            # versions SILO supports; newer releases only warn about its spelling.
            warnings.simplefilter("ignore")
            table.create_fts_index(
                "search_text", replace=True, use_tantivy=False,
                stem=True, remove_stop_words=True, ascii_folding=True, lower_case=True,
            )
            if table.count_rows() >= ANN_THRESHOLD:
                table.create_index(
                    metric="cosine", index_type="IVF_HNSW_SQ", vector_column_name="vector",
                    replace=True,
                )
            table.optimize()

    _COLUMNS = ["id", "doc_id", "chunk_index", "text", "search_text", "headings",
                "page_start", "page_end"]

    def vector_search(self, vector: np.ndarray, limit: int) -> list[dict]:
        table = self._open_table()
        if table is None:
            return []
        return (
            table.search(vector.tolist(), vector_column_name="vector")
            .distance_type("cosine").select([*self._COLUMNS, "_distance"]).limit(limit).to_list()
        )

    def keyword_search(self, query: str, limit: int) -> list[dict]:
        table = self._open_table()
        if table is None:
            return []
        try:
            return (
                table.search(query, query_type="fts", fts_columns="search_text")
                .select([*self._COLUMNS, "_score"]).limit(limit).to_list()
            )
        except Exception:
            # No keyword index yet, or the query reduced to nothing but stop words.
            return []

    def passages(self, doc_id: str, first: int, last: int) -> list[dict]:
        """Passages of one document by position, in reading order."""
        table = self._open_table()
        if table is None:
            return []
        rows = (
            table.search()
            .where(f"doc_id = {_sql_quote(doc_id)} AND chunk_index >= {int(first)} "
                   f"AND chunk_index <= {int(last)}")
            .select(self._COLUMNS).limit(max(1, int(last) - int(first) + 1)).to_list()
        )
        return sorted(rows, key=lambda r: r["chunk_index"])

    def size_bytes(self) -> int:
        root = self.settings.index_dir
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0

    def reset(self) -> None:
        import shutil

        self._writable()

        with self._lock:
            self._db.close()
            self._table = None
            self._lance = None
            shutil.rmtree(self.settings.index_dir, ignore_errors=True)
            self.settings.index_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.settings.catalogue_path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(SCHEMA)
            self._db.commit()
