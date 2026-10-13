# SILO

*Search and Indexing, Local-Only*

**Local-first document search that runs on the CPU you already have, with no network connection.**

Point SILO at a folder of PDFs, Word files, slides and spreadsheets. It parses them with
[Docling](https://github.com/docling-project/docling), indexes every passage for both keyword
and semantic search, and gives you a web interface where each result opens the source document
at the matching page. Nothing leaves the machine. An LLM is optional, and off by default.

![SILO search results with the source PDF open at the matching page](docs/images/silo-search.png)

A data silo is usually a complaint. For documents that must not leave the building, it is the requirement.

- **Air-gapped by construction.** One command (`silo models pull`) uses the network. Every
  other command forces the model libraries offline, so a missing model is a clear local error,
  never a silent download.
- **CPU-only.** Parsing, embedding and reranking all run on ordinary processors. No GPU, no
  CUDA, no drivers.
- **Hybrid retrieval.** BM25 keyword search and dense vector search run side by side, are fused
  by reciprocal rank, and the best candidates are reranked by a cross-encoder. Exact part
  numbers and loosely worded questions both work.
- **Every answer has an address.** Results carry the file, page and section they came from.
- **Optional intelligence.** Connect any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM,
  or a hosted provider) to get cited answers. SILO tells you whether that endpoint is on
  this machine, on your network, or external.
- **Portable index stores.** An index is one directory. Point SILO at any store with
  `--index`, share one read-only on a team drive, or carry one to another machine.
- **Works as tools for Claude.** A built-in MCP server lets Claude Desktop, Claude Code or any
  other MCP client search the index and read passages on your behalf.
- **Small and readable.** About 2,200 lines of Python and a dependency-free web page. No
  server database, no message queue, no build step.

## Quick start

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), about 2.5 GB of disk
(dependencies and models) and 4 GB of free memory. Developed and tested on macOS (Apple
Silicon); Linux and Windows use the same code path.

```bash
git clone https://github.com/parrytechnologies/silo.git
cd silo
uv sync                                   # install (CPU-only wheels)
uv run silo models pull                # one-time download, ~900 MB — the only online step
./scripts/fetch-sample-corpus.sh          # optional: six public PDFs (NIST frameworks, two papers)
uv run silo index sample-corpus        # or any folder of your own
uv run silo serve                      # http://127.0.0.1:8321
```

Prefer pip? `python -m venv .venv && . .venv/bin/activate && pip install -e .`
(on Linux add `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid CUDA wheels).

Search from the terminal too:

```console
$ uv run silo search "PR.AA-05" -k 1
 1 nist-cybersecurity-framework-2.0.pdf  1.00 · 4 matches
   /path/to/silo/sample-corpus/nist-cybersecurity-framework-2.0.pdf
   § Appendix A. CSF Core · p.20
   § PROTECT (PR): Safeguards to manage the organization's cybersecurity risks are used · p.24
   § PROTECT (PR): Safeguards to manage the organization's cybersecurity risks are used · p.25
   - o PR.AA-05: Access permissions, entitlements, and authorizations are defined in a
   policy, managed, enforced, and reviewed, and incorporate the principles of least …
```

Results are files, not passages: each one lists where in the file the query matched, says
if the file has changed since it was indexed, and names any other paths that hold an identical
copy. Add `--passages` for the individual matching passages.

## How it works

```
 folder ──▶ Docling ──▶ structure-aware ──▶ bge-small ──▶ LanceDB  (vectors + BM25 index)
            (layout,     chunker             embeddings     SQLite   (document catalogue)
             tables,     (≤384 tokens,       (ONNX, CPU)
             OCR opt.)    heading context)

 query ──▶ BM25 top 50 ───┐
       └─▶ vector top 50 ─┴─▶ reciprocal rank fusion ──▶ cross-encoder rerank (top 20) ──▶ results with
                                                                                            file, page, section
                                                                                            └─▶ optional LLM ──▶ cited answer
```

| Stage | Component | Why |
|---|---|---|
| Parsing | Docling (layout model + TableFormer, optional RapidOCR) | Recovers reading order, headings, tables and page numbers instead of a flat text dump |
| Chunking | Docling `HybridChunker`, sized with the embedding model's own tokenizer | Passages follow document structure and carry their heading path |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, quantised ONNX (67 MB) | Fast on CPU, MIT-licensed, widely used |
| Storage | LanceDB (embedded) + SQLite | One directory on disk. No database server to run, patch or secure |
| Keyword search | LanceDB native BM25 index | Exact identifiers, names, codes |
| Fusion | Reciprocal rank fusion, k = 60 | Combines rankings without tuning score scales |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX) | Reads query and passage together |
| Interface | FastAPI + one static page (no CDN, no external fonts) | Works with the network cable unplugged; a strict Content-Security-Policy enforces it |

More detail: [docs/architecture.md](docs/architecture.md).

## Measured, not promised

One run on an Apple M3 laptop (8 cores, 16 GB), inference device forced to `cpu`:

| | |
|---|---|
| Corpus | 6 PDFs, 227 pages, 12.8 MB |
| Indexing | 132 s end to end, about 1.7 pages per second, 544 passages |
| Peak memory while indexing | 3.3 GB |
| Index size on disk | 3.7 MB |
| Query, hybrid without rerank | about 15 ms |
| Query, hybrid with rerank | about 0.6–0.7 s |
| Re-index with nothing changed | under 1 s |

These are timings from one machine and one small corpus, not a benchmark. Retrieval quality has
been checked with the test suite and by hand, not against a labelled evaluation set. Run it on
your own documents before you rely on it.

## Proving the air gap

```bash
uv run silo doctor                               # models present, libraries offline, egress status
./scripts/verify-airgap-macos.sh sample-corpus      # macOS: index + search with outbound network denied
unshare -rn uv run silo index sample-corpus      # Linux: same idea, empty network namespace
```

To deploy on a machine that has never been online, run `silo models pull` elsewhere and copy
`~/.silo/models` across. See [docs/air-gapped.md](docs/air-gapped.md).

## Docker

```bash
docker build -t silo .
docker run --rm -v silo-home:/data silo models pull                      # online, once
docker run --rm --network none -v silo-home:/data -v /path/to/docs:/docs:ro \
    silo index /docs                                                        # no network at all
docker run --rm -p 127.0.0.1:8321:8321 -v silo-home:/data -v /path/to/docs:/docs:ro silo
```

The image is CPU-only (about 3 GB; no CUDA libraries) and runs as a non-root user. Indexing
and search have been run in it with `--network none`.

## Index stores: pick one, share one, move one

An index store is a single directory holding the passages, vectors, keyword index and a
catalogue. By default it lives at `~/.silo/index`, but nothing ties you to that one.

```bash
uv run silo index ~/contracts --index ~/stores/contracts      # build a separate store
uv run silo search "notice period" --index ~/stores/contracts # use it for one command
uv run silo serve --index /Volumes/team/silo-index --read-only   # serve a shared store, never write
uv run silo store use /Volumes/team/silo-index --read-only    # make it the default from now on
uv run silo store info                                        # what is this store?
uv run silo store relocate /srv/documents ~/Documents/contracts  # the files live elsewhere here
```

A store records the embedding model that built it, and SILO uses that model when it opens the
store, so an index built on one machine can be searched on another without matching settings
by hand. Search works even when the original files are not present; results then show the
indexed text instead of opening the file. Details and caveats:
[docs/index-stores.md](docs/index-stores.md).

## Use it from Claude (MCP)

`silo mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio.
The client gets six read-only tools: `find_sources`, `search_documents`, `get_passage_context`,
`read_document`, `list_documents` and `index_status`. `find_sources` returns the relevant
files with their absolute paths and matching sections, so a client that can read files opens
the current file instead of relying on the indexed text.

```bash
# Claude Code
claude mcp add silo -- /absolute/path/to/silo/.venv/bin/silo mcp
```

```jsonc
// Claude Desktop: Settings → Developer → Edit Config
{ "mcpServers": { "silo": { "command": "/absolute/path/to/silo/.venv/bin/silo", "args": ["mcp"] } } }
```

Then ask in plain language: *"Use silo to find what our policy says about carrying leave
forward, and cite the page."* Claude decides which searches to run, reads around the hits, and
answers with file and page references.

> **This changes where your text goes.** The MCP server makes no network connections, but the
> passages it returns are read by the client. With Claude that means they are sent to Anthropic
> as part of the conversation. For a fully air-gapped setup, use the built-in search, a local
> LLM, or an MCP client that runs a local model. See [docs/mcp.md](docs/mcp.md).

## Adding an LLM (optional)

```bash
export SILO_LLM_BASE_URL=http://localhost:11434/v1   # Ollama; any OpenAI-compatible URL works
export SILO_LLM_MODEL=gemma4:e2b
uv run silo ask "What are the tenets of zero trust?"
```

![A generated answer with numbered citations linking to source passages](docs/images/silo-answer.png)

The web interface gains an **Answer with LLM** switch and a status chip showing where prompts
go: loopback, private network, or external. Retrieval stays local either way; only the handful
of retrieved passages are sent to the model. See [docs/llm.md](docs/llm.md).

## Commands

| Command | Does |
|---|---|
| `silo models pull` / `models status` | Download or list local models |
| `silo index <folder> [--ocr] [--force] [--quiet]` | Index new and changed files; drop deleted ones |
| `silo search "<query>" [-k N] [--mode hybrid\|semantic\|keyword] [--passages] [--json]` | Search from the terminal: relevant files, or passages with `--passages` |
| `silo ask "<question>"` | Cited answer from the configured LLM |
| `silo serve [--host] [--port]` | Web interface and JSON API |
| `silo mcp` | MCP server on stdio for Claude and other MCP clients |
| `silo store info` / `store use <path>` / `store relocate <old> <new>` | Inspect, select and re-point index stores |
| `silo status` / `doctor` / `reset` | Inspect the index, check offline readiness, delete the index |

Every command that touches an index accepts `--index <path>`.

Formats: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, AsciiDoc, plain text; scanned PDFs and
images with `--ocr`. Settings: [docs/configuration.md](docs/configuration.md).

## What the engine deliberately leaves out

SILO is a single-user engine bound to `127.0.0.1`. It has **no authentication, no
per-document permissions and no multi-tenant isolation**. Those belong to the deployment around
it, and getting them wrong is how internal search tools leak documents.

| In this repository (Apache-2.0) | Deployment work, done per organisation |
|---|---|
| Parsing, chunking, embedding, hybrid search, reranking | Single sign-on and per-user, per-folder access control |
| Web interface and JSON API | Connectors and scheduled indexing for file shares, SharePoint, a DMS |
| Incremental indexing, OCR | Multi-user serving, TLS, audit logging, backups, monitoring |
| Optional LLM connector, MCP server (stdio, single user) | Sizing and hosting a local LLM; evaluation on your own documents |
| Index stores you can copy or share read-only | Per-team stores with access rules, kept current centrally |

[Parry Technologies](https://parrytech.co) builds that outer layer on an organisation's own
infrastructure. [docs/deployment.md](docs/deployment.md) describes how far the open engine goes
on its own (including cron and systemd examples) and where the service starts. To talk about a
deployment: [parryh@parrytech.co](mailto:parryh@parrytech.co).

## Development

```bash
uv sync
uv run pytest          # end-to-end tests use the real models if they are pulled
uv run ruff check src tests
```

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Models are downloaded on request and
governed by their own licences.
