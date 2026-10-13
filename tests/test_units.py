"""Fast tests that need no models."""

from __future__ import annotations

import os

import numpy as np

from silo import models
from silo.config import load_settings
from silo.indexer import discover
from silo.llm import build_messages, egress
from silo.search import (
    Hit,
    Searcher,
    SearchResult,
    group_sources,
    reading_order,
    source_state,
)
from silo.store import Store


def test_offline_is_the_default():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_settings_precedence(tmp_path, monkeypatch):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text('chunk_tokens = 256\nrerank = false\nllm_model = "a"\n')
    monkeypatch.setenv("SILO_LLM_MODEL", "b")
    s = load_settings(home)
    assert s.chunk_tokens == 256 and s.rerank is False
    assert s.llm_model == "b"  # env beats file
    assert not s.llm_enabled  # needs a base URL too


def test_missing_models_is_a_clear_error(settings):
    try:
        models.require(settings)
    except models.ModelsMissing as exc:
        assert "silo models pull" in str(exc)
    else:
        raise AssertionError("expected ModelsMissing")


def test_discover_skips_hidden_and_unsupported(corpus, settings):
    names = sorted(p.name for p in discover(corpus, settings))
    assert names == ["expenses.md", "leave-policy.pdf", "pump-maintenance.pdf"]


def test_egress_classification(settings):
    assert egress(settings) == "none"
    settings.llm_model = "m"
    for url, expected in [("http://localhost:11434/v1", "loopback"),
                          ("http://127.0.0.1:8080/v1", "loopback"),
                          ("http://10.0.4.12:8000/v1", "private"),
                          ("https://8.8.8.8/v1", "external")]:
        settings.llm_base_url = url
        assert egress(settings) == expected, url


def test_prompt_numbers_sources():
    hit = Hit(1, "d:0", "d", "T", "a/b.pdf", ".pdf", "The limit is five days.", "Leave", 2, 2, 1.0)
    system, user = build_messages("What is the limit?", [hit])
    assert "Use only the passages" in system["content"]
    assert "[1] a/b.pdf, page 2 — Leave" in user["content"]
    assert user["content"].endswith("Question: What is the limit?")


class _FakeEmbedder:
    dim = 2

    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeStore:
    def __init__(self):
        row = lambda i: {"id": f"d:{i}", "doc_id": "d", "chunk_index": i, "text": f"t{i}",  # noqa: E731
                         "search_text": f"t{i}", "headings": "", "page_start": i, "page_end": i}
        self.vec = [{**row(1), "_distance": 0.1}, {**row(2), "_distance": 0.2},
                    {**row(3), "_distance": 0.3}]
        self.kw = [{**row(3), "_score": 9.0}, {**row(4), "_score": 5.0}]

    def vector_search(self, vector, limit):
        return self.vec[:limit]

    def keyword_search(self, query, limit):
        return self.kw[:limit]

    docs = {"d": {"title": "Doc", "rel_path": "doc.pdf", "suffix": ".pdf", "path": "/x",
                  "sha256": "same", "mtime": 1.0, "size": 1}}

    def documents_by_ids(self, ids):
        return {i: self.docs[i] for i in ids if i in self.docs}

    def documents_by_sha(self, shas):
        return {sha: [d for d in self.docs.values() if d["sha256"] == sha] for sha in shas}


def test_reciprocal_rank_fusion(settings):
    searcher = Searcher(settings, _FakeStore(), _FakeEmbedder())
    result = searcher.search("anything", k=4, mode="hybrid", rerank=False)
    # Chunk 3 is the only one both retrievers found, so fusion puts it first.
    assert result.hits[0].chunk_id == "d:3" and result.hits[1].chunk_id == "d:1"
    assert {h.chunk_id for h in result.hits[2:]} == {"d:2", "d:4"}  # tied at rank 2
    top = result.hits[0]
    assert (top.vector_rank, top.keyword_rank) == (3, 1)
    assert abs(top.rrf - (1 / 63 + 1 / 61)) < 1e-9
    assert [h.chunk_id for h in searcher.search("x", mode="keyword", rerank=False).hits] == \
        ["d:3", "d:4"]
    assert searcher.search("   ", rerank=False).hits == []


def test_identical_files_are_returned_once(settings):
    store = _FakeStore()
    store.docs = {**store.docs, "e": {**store.docs["d"], "rel_path": "copy.pdf", "path": "/y"}}
    twin = {**store.vec[0], "id": "e:1", "doc_id": "e", "_distance": 0.15}
    store.vec = [store.vec[0], twin, *store.vec[1:]]
    hits = Searcher(settings, store, _FakeEmbedder()).search(
        "anything", k=10, mode="semantic", rerank=False).hits
    assert [h.chunk_id for h in hits] == ["d:1", "d:2", "d:3"]  # e:1 repeats d:1
    assert hits[0].also_at == ["/y"] and hits[0].path == "/x"
    assert hits[0].source_state == "missing" and hits[0].file_available is False


def test_sources_group_passages_by_file():
    def hit(rank, doc, index, score, headings, page):
        return Hit(rank, f"{doc}:{index}", doc, doc.upper(), f"{doc}.pdf", ".pdf",
                   f"text  of {doc} {index}", headings, page, page, score, path=f"/r/{doc}.pdf")

    result = SearchResult("q", "hybrid", True, [
        hit(1, "a", 9, 0.9, "Late", 4), hit(2, "b", 0, 0.8, "Only", 1),
        hit(3, "a", 2, 0.7, "Early", 1), hit(4, "a", 3, 0.6, "Early", 1),
        hit(5, "c", 0, 0.5, "", 0),
    ], {})
    grouped = group_sources(result, k=2)
    assert [(s.rank, s.doc_id, s.matches, s.score) for s in grouped.sources] == \
        [(1, "a", 3, 0.9), (2, "b", 1, 0.8)]
    first = grouped.sources[0]
    # One entry per section, in reading order, pointing at that section's best passage.
    assert [(s.headings, s.passage_id) for s in first.sections] == \
        [("Early", "a:2"), ("Late", "a:9")]
    assert first.path == "/r/a.pdf" and first.snippet == "text of a 9"
    assert grouped.to_dict()["sources"][1]["sections"][0]["page_start"] == 1


def test_source_state_tracks_the_file_on_disk(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("first")
    stat = note.stat()
    from silo.store import file_sha256
    doc = {"path": str(note), "mtime": stat.st_mtime, "size": stat.st_size,
           "sha256": file_sha256(note)}
    assert source_state(doc) == "current"
    os.utime(note, (stat.st_atime, stat.st_mtime + 60))  # touched, same content
    assert source_state(doc) == "current"
    note.write_text("second, and longer")
    assert source_state(doc) == "modified"
    note.unlink()
    assert source_state(doc) == "missing"


def test_catalogue_roundtrip_and_model_guard(settings):
    store = Store(settings)
    store.upsert_document(doc_id="a", path="/r/a.pdf", root="/r", rel_path="a.pdf", size=1,
                          mtime=1.0, sha256="x", status="indexed", pages=3, chunks=7)
    assert store.stats()["pages"] == 3 and store.roots() == ["/r"]
    store.check_embedding("model-a", 384)
    try:
        store.check_embedding("model-b", 384)
    except Exception as exc:
        assert "silo reset" in str(exc)
    else:
        raise AssertionError("expected IndexMismatch")


def test_reading_order_groups_by_document():
    def hit(rank, doc, index):
        return Hit(rank, f"{doc}:{index}", doc, "", f"{doc}.pdf", ".pdf", "", "", 1, 1, 1.0)

    result = SearchResult("q", "hybrid", True,
                          [hit(1, "b", 7), hit(2, "a", 3), hit(3, "b", 2), hit(4, "a", 1)], {})
    ordered = reading_order(result)
    assert [h.chunk_id for h in ordered.hits] == ["b:2", "b:7", "a:1", "a:3"]
    assert [h.rank for h in ordered.hits] == [1, 2, 3, 4]


def test_index_store_can_be_pointed_elsewhere(tmp_path, monkeypatch):
    home = tmp_path / "h"
    assert load_settings(home).index_dir == home.resolve() / "index"
    monkeypatch.setenv("SILO_INDEX", str(tmp_path / "shared" / "team-index"))
    monkeypatch.setenv("SILO_READ_ONLY", "1")
    settings = load_settings(home)
    assert settings.index_dir == (tmp_path / "shared" / "team-index").resolve()
    assert settings.read_only is True
    assert settings.models_dir == home.resolve() / "models"  # models stay with the machine


def test_store_use_writes_and_clears_config(tmp_path):
    from silo.cli import _write_config

    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text('ocr = true\nindex = "/old"\n')
    _write_config(home, {"index": "/srv/team index", "read_only": True})
    settings = load_settings(home)
    assert settings.index == "/srv/team index" and settings.read_only and settings.ocr
    _write_config(home, {"index": None, "read_only": None})
    settings = load_settings(home)
    assert settings.index == "" and not settings.read_only and settings.ocr
