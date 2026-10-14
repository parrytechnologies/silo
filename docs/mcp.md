# Using SILO from Claude and other MCP clients

`silo mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server. It lets an
assistant use your local index as a set of tools: it decides what to search for, finds the
files and sections that match, reads them, and answers with citations.

## What crosses the boundary

Read this first, because it is the one place SILO's air-gap promise depends on your choice of
client.

| Component | Network use |
|---|---|
| `silo mcp` (the server) | None. It talks to the client over stdin/stdout only. |
| The index store | Opened read-only. No tool can change or delete anything. |
| The client | Whatever the client does. **Claude Desktop and Claude Code send tool results, meaning the passages SILO returns, to Anthropic as part of the conversation.** |

So: with a cloud assistant, retrieval is local and the retrieved passages are not. That is
often exactly what is wanted (the assistant only ever sees the few passages it asks for, not
your document store). It is not air-gapped. For that, use the web interface, `silo ask` with a
local model, or an MCP client driving a local model.

## Setup

Use absolute paths; MCP clients do not start in your project directory.

**Claude Code**

```bash
claude mcp add silo -- /absolute/path/to/silo/.venv/bin/silo mcp
# a specific store, read by this server only:
claude mcp add silo-contracts -- /absolute/path/to/silo/.venv/bin/silo mcp --index /stores/contracts
```

**Claude Desktop** — Settings → Developer → Edit Config, then restart the app:

```json
{
  "mcpServers": {
    "silo": {
      "command": "/absolute/path/to/silo/.venv/bin/silo",
      "args": ["mcp"],
      "env": { "SILO_INDEX": "/absolute/path/to/an/index/store" }
    }
  }
}
```

Omit `env` to use the default store. Any other MCP client that can launch a stdio server is
configured the same way: command `silo`, argument `mcp`.

## Tools

All six are marked read-only.

| Tool | Arguments | Returns |
|---|---|---|
| `find_sources` | `query`, `limit` (1–50 files, default 8), `mode` | Ranked files, one entry each: absolute `path`, `state`, `also_at`, matched `sections` (heading, page, `passage_id`), `matches`, score and a short snippet |
| `search_documents` | `query`, `limit` (1–25, default 8), `mode` (`hybrid`, `semantic`, `keyword`) | Ranked passages with `passage_id`, `document_id`, file, page, section, score, text |
| `get_passage_context` | `passage_id`, `before` (default 2), `after` (default 2) | The passage and its neighbours in reading order |
| `read_document` | `document_id`, `start`, `count` (max 20) | Passages in order, with `next_start` for paging |
| `list_documents` | `contains`, `limit` | Document ids, files, titles, page and passage counts |
| `index_status` | — | Store location and id, counts, embedding model, source folders |

Passages are capped at 4,000 characters each. Every tool that names a document also returns
its absolute `path`.

## Sources first, passages second

The index stores the text of each passage as it was when the file was last indexed. It is a
copy, and it goes out of date when a file is edited and not re-indexed. `find_sources` is
built for clients that can read files themselves (Claude Code, for one): it says which files
are relevant and where in them to look, and the client reads the file on disk.

- `state` is `current` when the file matches what was indexed, `modified` when it has changed
  since, and `missing` when it is not at that path on this machine (an index used away from
  its documents). A `modified` file can still be read; its listed sections may have moved.
- `also_at` lists other paths holding a byte-identical copy of the file. Identical copies are
  returned once, by `find_sources` and by `search_documents`.
- A file ranks by its best matching passage. `matches` counts how many retrieved passages came
  from it and only breaks ties. Long lists are reranked at the top only: sources with
  `reranked: false` were ordered by rank fusion and follow the reranked ones.
- The web interface lists the same sources, 50 at a time, with each file's matching passages
  underneath. The default here is 8 because every source costs the client context.
- Sections are headings and page numbers, not line numbers.

Use `search_documents`, `get_passage_context` and `read_document` when the client cannot read
files, or when the source is a long PDF or office document and the matching passages are
enough. A file that was created or edited after the last `silo index` run is found only by
its old text, or not at all, so re-index on a schedule.

## What to ask

- *"Use silo to find what the supplier agreement says about liability caps. Quote the clause
  and give the page."*
- *"Search silo for every mention of PR.AA-05 and summarise what each document requires."*
- *"List the documents in silo, then read the introduction of the zero trust paper and
  explain its scope."*

Assistants do better with several narrow searches than one broad one, and the server's
instructions tell them so. Exact identifiers work best in `keyword` mode.

## Checked

- The test suite starts `silo mcp` as a subprocess and drives it with the official MCP client:
  tool listing, sources, search, context, paged reading, status and error handling.
- Claude Code, run headless with only the SILO tools allowed, answered a question about a
  control identifier in the sample corpus with the correct file and page, using one
  `search_documents` call in keyword mode.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Client shows no tools | Wrong path to `silo`, or the virtual environment was moved. Run the command by hand: it should wait silently for input. |
| "No index store at …" | Nothing indexed yet, or `SILO_INDEX` points at the wrong directory. |
| "Models not found …" | Run `silo models pull` on this machine. |
| First search takes a couple of seconds | Models load on first use, then stay loaded. |
