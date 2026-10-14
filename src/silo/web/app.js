"use strict";
// SILO web interface. No frameworks, no build step, no external requests.
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const state = { status: null, hits: [], query: "", active: -1, reader: null, controller: null, poll: null };
const FILES = 50; // files listed per search
const SHOWN = 3; // matching passages listed per file before "show all"
const STEP = 5; // passages added by "earlier" / "later" in the reader
const EXAMPLES = ["What does the policy say about third-party risk?", "termination notice period", "Who is responsible for access reviews?"];
const STOP = new Set("a an and are as at be by for from how in is it of on or that the this to was what when where which who why with do does should can".split(" "));

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) { let detail = response.statusText; try { detail = (await response.json()).detail || detail; } catch (_) {} throw new Error(detail); }
  return response;
}

function chip(text, cls) { const c = el("span", "chip" + (cls ? " " + cls : "")); c.append(el("i"), document.createTextNode(text)); return c; }

async function loadStatus() {
  const s = state.status = await (await api("/api/status")).json();
  const chips = $("chips"); chips.replaceChildren();
  const egress = s.llm.egress;
  if (egress === "external") chips.append(chip("LLM: external endpoint", "warn keep"));
  else if (egress === "none") chips.append(chip(s.offline ? "Offline · no egress" : "No LLM · no egress", "keep"));
  else chips.append(chip("LLM on " + egress + " network", "keep"));
  if (s.index.read_only) chips.append(chip("Read-only index"));
  chips.append(chip("CPU · " + s.models.threads + " threads"), chip(s.stats.indexed + " docs · " + s.stats.chunks + " passages"));
  $("ask-switch").hidden = !s.llm.enabled;
  $("foot-models").textContent = "Embeddings " + s.models.embedding + (s.models.reranker ? " · Reranker " + s.models.reranker : "") + (s.llm.enabled ? " · LLM " + s.llm.model : "") + " · v" + s.version;
  renderJob(s.job);
  if (!state.hits.length && !$("q").value) renderEmpty();
  return s;
}

function renderEmpty(message) {
  const box = $("empty"); box.replaceChildren(); box.hidden = false;
  const s = state.status;
  if (message) { box.append(el("h2", null, "No matching passages."), el("p", null, message)); return; }
  if (s && s.stats.chunks === 0) {
    box.append(el("h2", null, "The index is empty."));
    const p = el("p"); p.append("Point SILO at a folder from a terminal, then come back: ", el("code", null, "silo index /path/to/documents")); box.append(p); return;
  }
  box.append(el("h2", null, "Ready when you are."), el("p", null, "Hybrid mode runs a keyword (BM25) search and a semantic vector search side by side, fuses the two rankings, then reranks the best candidates. Results are files; each lists the passages that matched and opens at the page they came from."));
  const row = el("div", "examples");
  EXAMPLES.forEach((q) => { const b = el("button", null, q); b.type = "button"; b.onclick = () => { $("q").value = q; runSearch(); }; row.append(b); });
  box.append(row);
}

function highlight(target, text, query) {
  const terms = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}.\-_]*/gu) || [])].filter((t) => t.length > 2 && !STOP.has(t));
  if (!terms.length) { target.textContent = text; return; }
  const pattern = new RegExp("(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "giu");
  text.split(pattern).forEach((part, i) => target.append(i % 2 ? el("mark", null, part) : document.createTextNode(part)));
}

const pageLabel = (h) => !h.page_start ? "" : h.page_end > h.page_start ? "pp. " + h.page_start + "–" + h.page_end : "p. " + h.page_start;
const fileUrl = (h) => "/api/documents/" + encodeURIComponent(h.doc_id) + "/file" + (h.suffix === ".pdf" && h.page_start ? "#page=" + h.page_start : "");

const position = (h) => Number(h.chunk_id.slice(h.chunk_id.lastIndexOf(":") + 1));

function gaugeFor(h, mode) {
  const gauge = el("div", "gauge");
  if (h.rerank != null) gauge.append(el("span", "lead", "rerank " + h.rerank.toFixed(3)));
  if (h.vector_rank) gauge.append(el("span", null, "vector #" + h.vector_rank + " · cos " + h.vector_similarity.toFixed(2)));
  if (h.keyword_rank) gauge.append(el("span", null, "bm25 #" + h.keyword_rank + " · " + h.bm25.toFixed(1)));
  if (h.rrf != null && mode === "hybrid") gauge.append(el("span", null, "rrf " + h.rrf.toFixed(4)));
  return gauge;
}

// One entry per file. Its matching passages are listed underneath in reading order and
// numbered across the whole list, so state.hits and selectHit work as they do for passages.
function renderSources(result) {
  state.hits = []; state.query = result.query; state.active = -1;
  const list = $("results"); list.replaceChildren();
  $("empty").hidden = result.sources.length > 0;
  if (!result.sources.length) renderEmpty("Try different wording, or switch retrieval mode. Keyword mode needs the exact term; semantic mode matches meaning.");
  result.sources.forEach((s, n) => {
    const li = el("li", "source");
    li.append(el("span", "hit-n", String(n + 1).padStart(2, "0")));
    const body = el("div");
    const title = el("h3", "hit-title"); const open = el("button", null, s.title || s.rel_path); open.type = "button"; title.append(open);
    const meta = el("div", "hit-meta");
    meta.append(el("span", null, s.rel_path), el("span", "count", s.matches + (s.matches === 1 ? " passage" : " passages")));
    if (s.source_state === "modified") meta.append(el("span", "badge warn", "Changed since indexing"));
    if (s.source_state === "missing") meta.append(el("span", "badge off", "File not on this machine"));
    if (s.also_at.length) { const copies = el("span", "badge", "+" + s.also_at.length + (s.also_at.length === 1 ? " identical copy" : " identical copies")); copies.title = s.also_at.join("\n"); meta.append(copies); }
    const rows = el("ol", "passages");
    const byRank = [...s.passages].sort((a, b) => a.rank - b.rank);
    const shown = new Set(byRank.slice(0, SHOWN));
    let best = 0;
    s.passages.forEach((h) => {
      const i = state.hits.push(h) - 1; if (h === byRank[0]) best = i;
      const row = el("li", "passage" + (shown.has(h) ? "" : " is-extra")); row.id = "hit-" + (i + 1);
      const jump = el("button"); jump.type = "button";
      const where = el("span", "where"); if (h.page_start) where.append(el("span", "page", pageLabel(h)));
      where.append(el("span", "crumb", h.headings || "Passage " + (position(h) + 1)));
      const text = el("span", "passage-text"); highlight(text, h.text, result.query);
      jump.append(where, text); jump.onclick = () => selectHit(i, false);
      row.append(jump, gaugeFor(h, result.mode)); rows.append(row);
    });
    body.append(title, meta, rows);
    if (s.passages.length > shown.size) { const more = el("button", "more", "Show all " + s.passages.length + " matching passages"); more.type = "button"; more.onclick = () => li.classList.add("is-open"); body.append(more); }
    open.onclick = () => selectHit(best, true);
    li.append(body); list.append(li);
  });
  const t = result.timings_ms, files = result.sources.length;
  $("readout").textContent = files ? files + (files === 1 ? " file · " : " files · ") + state.hits.length + " passages · " + Object.entries(t).map(([k, v]) => k + " " + Math.round(v) + "ms").join(" · ") : "";
  if (files) selectHit(0, false);
}

function renderHits(result) {
  state.hits = result.hits; state.query = result.query; state.active = -1;
  const list = $("results"); list.replaceChildren();
  $("empty").hidden = result.hits.length > 0;
  if (!result.hits.length) renderEmpty("Try different wording, or switch retrieval mode. Keyword mode needs the exact term; semantic mode matches meaning.");
  result.hits.forEach((h, i) => {
    const li = el("li", "hit"); li.id = "hit-" + (i + 1);
    li.append(el("span", "hit-n", String(i + 1).padStart(2, "0")));
    const body = el("div");
    const title = el("h3", "hit-title"); const open = el("button", null, h.title || h.rel_path); open.type = "button"; title.append(open);
    const meta = el("div", "hit-meta");
    if (h.page_start) meta.append(el("span", "page", pageLabel(h)));
    meta.append(el("span", null, h.rel_path));
    if (h.headings) meta.append(el("span", "crumb", h.headings));
    const text = el("p", "hit-text"); highlight(text, h.text, result.query);
    body.append(title, meta, text, gaugeFor(h, result.mode)); li.append(body);
    li.onclick = (e) => { if (!window.getSelection().toString()) selectHit(i, e.target.closest("button") != null); };
    list.append(li);
  });
  const t = result.timings_ms;
  $("readout").textContent = result.hits.length ? result.hits.length + " results · " + Object.entries(t).map(([k, v]) => k + " " + Math.round(v) + "ms").join(" · ") : "";
  if (result.hits.length) selectHit(0, false);
}

function selectHit(i, fromKeyboardOrTitle) {
  const h = state.hits[i]; if (!h) return;
  document.querySelectorAll(".hit.is-active, .passage.is-active, .source.is-active").forEach((n) => n.classList.remove("is-active"));
  const node = $("hit-" + (i + 1)); node.classList.add("is-active"); state.active = i; state.reader = null;
  const source = node.closest(".source");
  if (source) { source.classList.add("is-active"); if (node.classList.contains("is-extra")) source.classList.add("is-open"); }
  const url = fileUrl(h);
  if (getComputedStyle($("preview")).display === "none") { if (fromKeyboardOrTitle && h.file_available) window.open(url, "_blank", "noopener"); return; }
  $("preview-label").textContent = h.rel_path + (h.page_start ? " · " + pageLabel(h) : "");
  const link = $("preview-open"); link.href = url; link.hidden = !h.file_available;
  const body = $("preview-body"); body.replaceChildren();
  if (h.file_available && h.suffix === ".pdf") { const frame = el("iframe"); frame.title = "Source document: " + h.rel_path; frame.src = url; body.append(frame); return; }
  const at = position(h);
  const note = !h.file_available ? "The source file is not on this machine, so this is the indexed text."
    : h.source_state === "modified" ? "This file has changed since it was indexed. This is the indexed text; open the original for the current version." : "";
  drawReader(state.reader = { hit: h, first: Math.max(0, at - 2), last: at + 2, active: at, note }, true);
}

// The indexed passages of one document around the selected one, extendable both ways.
async function drawReader(reader, toActive) {
  const h = reader.hit, body = $("preview-body"), box = el("div", "reader");
  let data;
  try { data = await (await api("/api/documents/" + encodeURIComponent(h.doc_id) + "/passages?start=" + reader.first + "&count=" + (reader.last - reader.first + 1))).json(); }
  catch (_) { data = { total: 0, passages: [{ position: reader.active, headings: h.headings, text: h.text }] }; }
  if (state.reader !== reader) return;
  if (reader.note) box.append(el("p", "reader-note", reader.note));
  const step = (label, change) => { const b = el("button", "more", label); b.type = "button"; b.onclick = () => { change(); drawReader(reader, false); }; return b; };
  if (reader.first > 0) box.append(step("↑ Earlier passages", () => { reader.first = Math.max(0, reader.first - STEP); }));
  let heading = null, active = null;
  data.passages.forEach((p) => {
    if (p.headings && p.headings !== heading) box.append(el("h4", "reader-head", p.headings));
    heading = p.headings;
    const para = el("p", "reader-passage" + (p.position === reader.active ? " is-active" : "")); highlight(para, p.text, state.query);
    if (p.position === reader.active) active = para;
    box.append(para);
  });
  if (reader.last < data.total - 1) box.append(step("↓ Later passages", () => { reader.last += STEP; }));
  const kept = body.firstChild ? body.firstChild.scrollTop : 0;
  body.replaceChildren(box);
  box.scrollTop = toActive && active ? Math.max(0, active.offsetTop - box.offsetTop - 60) : kept;
}

const currentMode = () => document.querySelector('input[name="mode"]:checked').value;

async function runSearch() {
  const q = $("q").value.trim(); if (!q) return;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  history.replaceState(null, "", "?q=" + encodeURIComponent(q) + "&mode=" + currentMode());
  showView("search"); document.body.classList.add("compact");
  $("readout").textContent = "searching…"; $("answer").hidden = true; if (!state.hits.length) $("empty").hidden = true;
  try {
    if ($("ask").checked && state.status && state.status.llm.enabled) return await runAsk(q);
    const params = new URLSearchParams({ q, mode: currentMode(), rerank: $("rerank").checked, k: FILES });
    const result = await (await api("/api/sources?" + params, { signal: state.controller.signal })).json();
    renderSources(result);
  } catch (err) { if (err.name !== "AbortError") { $("readout").textContent = ""; $("results").replaceChildren(); renderEmpty(err.message); } }
}

function renderAnswerText(target, text) {
  target.replaceChildren();
  text.split(/(\[\d+\])/g).forEach((part) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (m && state.hits[m[1] - 1]) { const b = el("button", "cite", m[1]); b.type = "button"; b.title = "Show source " + m[1]; b.onclick = () => { selectHit(m[1] - 1, false); $("hit-" + m[1]).scrollIntoView({ block: "center", behavior: "smooth" }); }; target.append(b); }
    else target.append(document.createTextNode(part));
  });
}

async function runAsk(question) {
  const box = $("answer"), body = $("answer-body"); box.hidden = false; body.replaceChildren(); body.classList.add("is-streaming");
  $("answer-meta").textContent = state.status.llm.model + " · " + state.status.llm.egress;
  const response = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, mode: currentMode() }), signal: state.controller.signal });
  const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "", text = "";
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
      const event = (/^event: (.*)$/m.exec(block) || [])[1], data = JSON.parse((/^data: (.*)$/m.exec(block) || [, "{}"])[1]);
      if (event === "sources") renderHits(data);
      else if (event === "token") { text += data.text; renderAnswerText(body, text); }
      else if (event === "error") body.append(el("p", "error", data.message));
      else if (event === "done" && data.seconds != null) $("answer-meta").textContent += " · " + data.seconds + "s";
    }
  }
  body.classList.remove("is-streaming");
}

function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
  $("view-search").hidden = name !== "search"; $("view-library").hidden = name !== "library";
  if (name === "library") loadLibrary();
}

const fmtBytes = (n) => n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";

async function loadLibrary() {
  const [s, d] = await Promise.all([loadStatus(), api("/api/documents").then((r) => r.json())]);
  const stats = $("stats"); stats.replaceChildren();
  [["Documents", s.stats.docs], ["Pages", s.stats.pages], ["Passages", s.stats.chunks], ["Source size", fmtBytes(s.stats.bytes)]].forEach(([k, v]) => { const div = el("div"); div.append(el("dt", null, k), el("dd", null, String(v))); stats.append(div); });
  const line = $("store-line"); line.replaceChildren();
  line.append("Index store ", el("code", null, s.index.path), " · " + fmtBytes(s.index.size_bytes) + (s.index.store_id ? " · id " + s.index.store_id : "") + (s.index.read_only ? " · read-only" : ""));
  $("rescan").hidden = s.index.read_only;
  const body = $("docs"); body.replaceChildren();
  d.documents.forEach((doc) => {
    const tr = el("tr"), name = el("td"), a = el(doc.file_available ? "a" : "strong", null, doc.title || doc.rel_path);
    if (doc.file_available) { a.href = "/api/documents/" + encodeURIComponent(doc.doc_id) + "/file"; a.target = "_blank"; a.rel = "noopener"; }
    name.append(a, el("small", null, doc.rel_path + " · " + fmtBytes(doc.size) + (doc.file_available ? "" : " · source file not on this machine") + (doc.error ? " · " + doc.error : "")));
    const status = el("td"); status.append(el("span", "status " + doc.status, doc.status));
    tr.append(name, el("td", null, doc.pages || "—"), el("td", null, String(doc.chunks)), el("td", null, doc.parse_seconds.toFixed(1) + " s"), status); body.append(tr);
  });
  if (!d.documents.length) { const tr = el("tr"), td = el("td", null, "Nothing indexed yet. Run: silo index /path/to/documents"); td.colSpan = 5; tr.append(td); body.append(tr); }
}

function renderJob(job) {
  const label = $("job"), button = $("rescan"); button.disabled = job.running;
  if (job.running) label.textContent = "Scanning " + job.done + "/" + job.total + (job.current ? " · " + job.current : "");
  else if (job.error) label.textContent = "Scan failed: " + job.error;
  else if (job.last) label.textContent = "Last scan: " + job.last.indexed + " indexed · " + job.last.unchanged + " unchanged · " + job.last.removed + " removed · " + job.last.failed + " failed";
  else label.textContent = "";
  if (job.running && !state.poll) state.poll = setInterval(async () => { const s = await loadStatus(); if (!s.job.running) { clearInterval(state.poll); state.poll = null; loadLibrary(); } }, 1500);
}

$("search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", runSearch));
$("rerank").addEventListener("change", runSearch);
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
$("rescan").addEventListener("click", async () => { try { await api("/api/index", { method: "POST" }); } catch (err) { $("job").textContent = err.message; return; } loadStatus(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("q")) { e.preventDefault(); $("q").focus(); $("q").select(); }
  if ((e.key === "ArrowDown" || e.key === "ArrowUp") && document.activeElement !== $("q") && state.hits.length) { e.preventDefault(); const next = Math.min(state.hits.length - 1, Math.max(0, state.active + (e.key === "ArrowDown" ? 1 : -1))); selectHit(next, false); $("hit-" + (next + 1)).scrollIntoView({ block: "nearest" }); }
});

(async function init() {
  const params = new URLSearchParams(location.search);
  if (params.get("mode")) { const r = document.querySelector('input[name="mode"][value="' + CSS.escape(params.get("mode")) + '"]'); if (r) r.checked = true; }
  await loadStatus().catch((err) => renderEmpty(err.message));
  if (params.get("q")) { $("q").value = params.get("q"); runSearch(); }
})();
