"""Index real files with the real CPU models, then search them. Skipped without models."""

from __future__ import annotations

import os
import shutil

import pytest
from fastapi.testclient import TestClient

from silo.embed import Embedder
from silo.indexer import Indexer
from silo.search import Searcher
from silo.server import create_app
from silo.store import Store


@pytest.fixture(scope="module")
def engine(real_settings, corpus, tmp_path_factory):
    work = tmp_path_factory.mktemp("docs")
    shutil.copytree(corpus, work, dirs_exist_ok=True)
    store = Store(real_settings)
    embedder = Embedder(real_settings)
    indexer = Indexer(real_settings, store, embedder)
    report = indexer.index(work)
    return real_settings, store, indexer, Searcher(real_settings, store, embedder), work, report


def test_index_report(engine):
    _, store, _, _, _, report = engine
    assert (report.total, report.indexed, report.failed) == (3, 3, 0)
    assert store.stats()["pages"] == 4  # two 2-page PDFs; markdown has no pages
    assert store.chunk_count() == report.chunks > 0


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "keyword"])
def test_finds_the_right_page(engine, mode):
    searcher = engine[3]
    hit = searcher.search("how many unused days can be carried forward", k=3, mode=mode).hits[0]
    assert hit.rel_path == "leave-policy.pdf" and hit.page_start == 2
    assert "five unused days" in hit.text


def test_semantic_match_without_shared_keywords(engine):
    # "holiday allowance" never appears in the corpus; "annual leave" does.
    hit = engine[3].search("holiday allowance for staff", k=3, mode="semantic").hits[0]
    assert hit.rel_path == "leave-policy.pdf" and hit.page_start == 1


def test_exact_identifier_lookup(engine):
    hit = engine[3].search("QX-7731-B", k=3, mode="hybrid").hits[0]
    assert hit.rel_path == "pump-maintenance.pdf" and hit.page_start == 2
    assert hit.keyword_rank == 1


def test_markdown_and_heading_context(engine):
    hit = engine[3].search("overnight meal allowance", k=3).hits[0]
    assert hit.rel_path == os.path.join("notes", "expenses.md")
    assert hit.page_start == 0 and "Meals" in hit.headings


def test_incremental_update_and_removal(engine):
    _, store, indexer, searcher, work, _ = engine
    again = indexer.index(work)
    assert (again.indexed, again.unchanged) == (0, 3)

    target = work / "notes" / "expenses.md"
    target.write_text(target.read_text() + "\n## Mileage\n\nPrivate car mileage is paid at 45 "
                      "pence per mile.\n")
    (work / "pump-maintenance.pdf").unlink()
    changed = indexer.index(work)
    assert (changed.indexed, changed.unchanged, changed.removed) == (1, 1, 1)
    assert searcher.search("mileage rate per mile", k=1).hits[0].rel_path.endswith("expenses.md")
    assert all(h.rel_path != "pump-maintenance.pdf"
               for h in searcher.search("QX-7731-B", k=5).hits)
    assert store.chunk_count() == store.stats()["chunks"]


def test_http_api(engine):
    settings = engine[0]
    client = TestClient(create_app(settings))
    status = client.get("/api/status").json()
    assert status["offline"] is True and status["llm"]["egress"] == "none"
    assert status["index"]["read_only"] is False and status["index"]["store_id"]
    assert status["models"]["device"] == "cpu"

    page = client.get("/")
    assert page.status_code == 200 and "SILO" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert "http://" not in page.text.replace("https://parrytech.co", "")

    body = client.get("/api/search", params={"q": "carry forward unused days", "k": 2}).json()
    top = body["hits"][0]
    assert top["rel_path"] == "leave-policy.pdf" and body["reranked"] is True
    file = client.get(f"/api/documents/{top['doc_id']}/file")
    assert file.status_code == 200 and file.content.startswith(b"%PDF")
    assert client.get("/api/documents/..%2F..%2Fetc%2Fpasswd/file").status_code == 404
    assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400

    listed = client.get("/api/sources", params={"q": "carry forward unused days"}).json()
    first = listed["sources"][0]
    assert first["rel_path"] == "leave-policy.pdf" and first["source_state"] == "current"
    assert len({s["doc_id"] for s in listed["sources"]}) == len(listed["sources"])
    assert any(p["page_start"] == 2 for p in first["passages"])
    order = [int(p["chunk_id"].rsplit(":", 1)[1]) for p in first["passages"]]
    assert order == sorted(order)  # reading order
    assert client.get("/api/sources", params={"q": "x", "k": 51}).status_code == 422
    reading = client.get(f"/api/documents/{first['doc_id']}/passages", params={"count": 1}).json()
    assert reading["total"] >= 1 and reading["passages"][0]["position"] == 0
    assert reading["passages"][0]["text"]
    assert client.get("/api/documents/nope/passages").status_code == 404
    assert client.post("/api/ask", json={"question": "anything"}).status_code == 409


def test_store_can_be_copied_relocated_and_opened_read_only(engine, tmp_path):
    """Take an index to another location, away from its documents, and still search it."""
    from silo.config import load_settings
    from silo.engine import open_engine
    from silo.store import ReadOnlyStore, Store

    settings, _, _, _, work, _ = engine
    copy = tmp_path / "borrowed-index"
    shutil.copytree(settings.index_dir, copy)

    other = load_settings(settings.home)
    other.index = str(copy)
    other.embedding_model = "some/other-model"  # the store, not local config, decides
    borrowed = open_engine(other, read_only=True)
    assert other.embedding_model == settings.embedding_model and borrowed.notices
    hit = borrowed.searcher.search("carry forward unused days", k=1).hits[0]
    assert hit.rel_path == "leave-policy.pdf" and hit.file_available is True
    with pytest.raises(ReadOnlyStore):
        borrowed.store.set_meta("x", "y")

    # Pretend the documents live elsewhere on the new machine.
    writable = Store(other, read_only=False)
    assert writable.relocate(str(work), str(tmp_path / "elsewhere")) == 2
    moved = open_engine(other, read_only=True)
    hit = moved.searcher.search("carry forward unused days", k=1).hits[0]
    assert hit.file_available is False and "five unused days" in hit.text
    assert moved.store.manifest()["store_id"] == engine[1].manifest()["store_id"]

    client = TestClient(create_app(other))
    status = client.get("/api/status").json()
    assert status["index"]["path"] == str(copy)
    assert client.get(f"/api/documents/{hit.doc_id}/file").status_code == 404
    other.read_only = True
    assert TestClient(create_app(other)).post("/api/index").status_code == 403


def test_missing_store_read_only_is_a_clear_error(real_settings, tmp_path):
    from silo.config import load_settings
    from silo.store import Store, StoreNotFound

    settings = load_settings(real_settings.home)
    settings.index = str(tmp_path / "nothing-here")
    with pytest.raises(StoreNotFound, match="silo index"):
        Store(settings, read_only=True)
    assert not (tmp_path / "nothing-here").exists()  # opening read-only creates nothing


def test_mcp_server_over_stdio(engine):
    """Drive the real `silo mcp` process the way an MCP client does."""
    import asyncio
    import json
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    settings = engine[0]
    env = {**os.environ, "SILO_HOME": str(settings.home), "SILO_INDEX": str(settings.index_dir)}

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "silo.cli", "mcp"],
                                       env=env)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            names = {tool.name for tool in (await client.list_tools()).tools}

            async def call(name, **arguments):
                result = await client.call_tool(name, arguments)
                failed = getattr(result, "is_error", getattr(result, "isError", False))
                return failed, result.content[0].text

            _, text = await call("search_documents", query="carry forward unused days", limit=2)
            found = json.loads(text)["results"]
            _, text = await call("find_sources", query="carry forward unused days", limit=2)
            sources = json.loads(text)["sources"]
            _, text = await call("get_passage_context", passage_id=found[0]["passage_id"])
            context = json.loads(text)
            _, text = await call("read_document", document_id=found[0]["document_id"], count=1)
            reading = json.loads(text)
            _, text = await call("index_status")
            status = json.loads(text)
            failed, message = await call("get_passage_context", passage_id="nonsense")
            return names, found, sources, context, reading, status, failed, message

    names, found, sources, context, reading, status, failed, message = asyncio.run(session())
    assert names == {"find_sources", "search_documents", "get_passage_context", "read_document",
                     "list_documents", "index_status"}
    assert found[0]["file"] == "leave-policy.pdf" and found[0]["page"] == 2
    top = sources[0]
    assert top["file"] == "leave-policy.pdf" and top["state"] == "current"
    assert os.path.isabs(top["path"]) and os.path.isfile(top["path"])
    assert top["path"] == found[0]["path"] == reading["path"]
    assert any(s.get("page") == 2 for s in top["sections"]) and "text" not in top
    assert len({s["file"] for s in sources}) == len(sources)  # one entry per file
    assert any("five unused days" in p["text"] for p in context["passages"])
    assert reading["total_passages"] >= 1 and reading["passages"][0]["position"] == 0
    assert status["passages"] > 0 and status["store_id"]
    assert failed and "passage_id should look like" in message


def test_sources_collapse_copies_and_flag_changed_files(engine):
    _, _, indexer, searcher, work, _ = engine
    (work / "archive").mkdir()
    shutil.copy(work / "leave-policy.pdf", work / "archive" / "leave-policy-copy.pdf")
    indexer.index(work)

    sources = searcher.sources("carry forward unused days", k=5).sources
    policies = [s for s in sources if "leave-policy" in s.rel_path]
    assert len(policies) == 1 and sources[0] is policies[0]
    top = policies[0]
    copy = work / "archive" / "leave-policy-copy.pdf"
    assert {top.path, *top.also_at} == {str(work / "leave-policy.pdf"), str(copy)}
    assert top.source_state == "current" and any(s.page_start == 2 for s in top.sections)
    hits = searcher.search("carry forward unused days", k=10).hits
    positions = [h.chunk_id.rsplit(":", 1)[1] for h in hits if "leave-policy" in h.rel_path]
    assert positions and len(set(positions)) == len(positions)  # each passage once, not twice

    note = work / "notes" / "expenses.md"
    note.write_text(note.read_text() + "\nEdited after indexing.\n")
    edited = searcher.sources("overnight meal allowance", k=1).sources[0]
    assert edited.rel_path.endswith("expenses.md") and edited.source_state == "modified"
