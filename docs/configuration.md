# Configuration

Precedence: built-in defaults → `$SILO_HOME/config.toml` → `SILO_*` environment
variables → command-line flags.

| Key (`config.toml`) | Env var | Default | Meaning |
|---|---|---|---|
| — | `SILO_HOME` | `~/.silo` | Models and index directory |
| `index` | `SILO_INDEX` | `<home>/index` | Index store to use (`--index` on the command line) |
| `read_only` | `SILO_READ_ONLY` | `false` | Never write to the index store |
| `embedding_model` | `SILO_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed text model |
| `reranker_model` | `SILO_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Any fastembed cross-encoder |
| `rerank` | `SILO_RERANK` | `true` | Rerank by default |
| `ocr` | `SILO_OCR` | `false` | OCR scanned pages and images |
| `tables` | `SILO_TABLES` | `true` | Recover table structure |
| `table_mode` | `SILO_TABLE_MODE` | `fast` | `fast` or `accurate` |
| `threads` | `SILO_THREADS` | `0` (all cores) | CPU threads for inference |
| `document_timeout` | `SILO_DOCUMENT_TIMEOUT` | `600` | Seconds allowed per document |
| `chunk_tokens` | `SILO_CHUNK_TOKENS` | `384` | Maximum passage length |
| `candidates` | `SILO_CANDIDATES` | `50` | Candidates per retriever before fusion |
| `rerank_top` | `SILO_RERANK_TOP` | `20` | Fused candidates sent to the reranker |
| `rrf_k` | `SILO_RRF_K` | `60` | Reciprocal rank fusion constant |
| `llm_base_url` | `SILO_LLM_BASE_URL` | empty (off) | OpenAI-compatible base URL |
| `llm_model` | `SILO_LLM_MODEL` | empty (off) | Model name |
| `llm_api_key` | `SILO_LLM_API_KEY` | empty | Bearer token if required |
| `llm_context_chunks` | `SILO_LLM_CONTEXT_CHUNKS` | `8` | Passages sent to the LLM |
| `host` / `port` | `SILO_HOST` / `SILO_PORT` | `127.0.0.1` / `8321` | Server bind |

Example `~/.silo/config.toml`:

```toml
ocr = true
threads = 6
llm_base_url = "http://localhost:11434/v1"
llm_model = "gemma4:e2b"
```

`embedding_model` and `chunk_tokens` are recorded in each index store and the store's values
win when it is opened. To change them, build a new store (or `silo reset` and re-index).

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Index statistics, models, LLM egress, indexing job state |
| GET | `/api/sources?q=&mode=&k=&rerank=` | Search by file (up to 50); each source has its path, state, identical copies and matching passages in reading order |
| GET | `/api/search?q=&mode=&k=&rerank=` | Search by passage; returns hits with ranks, scores and timings |
| GET | `/api/documents/{doc_id}/passages?start=&count=` | A document's indexed passages in order |
| GET | `/api/documents` | Catalogue listing |
| GET | `/api/documents/{id}/file` | The original file (only catalogued files are served) |
| POST | `/api/ask` `{question, mode}` | Server-sent events: `sources`, `token`…, `done` |
| POST | `/api/index` | Re-scan every indexed folder in the background (403 on a read-only store) |
| GET | `/api/openapi.json` | Machine-readable schema |
