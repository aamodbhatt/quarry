"use strict";

// quarry web UI: streams agent steps over SSE and renders plan, SQL, result, chart and trace.
// Everything from the server (model text, database values) is untrusted: it is escaped or
// set through textContent, never injected as HTML.

const $ = (id) => document.getElementById(id);
const state = { thread: newThread(), running: false, sqlCards: [], source: null };

function newThread() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------------------------------------------------------------- SQL highlighting

const KEYWORDS = new Set(("select from where join left right inner outer cross full on group by order " +
  "having limit offset as and or not in is null case when then else end with distinct union all " +
  "intersect except asc desc between like glob exists cast over partition recursive insert into " +
  "update delete drop create alter table values set attach pragma").split(" "));

function highlightSql(sql) {
  const token = /(--[^\n]*)|('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(\s*\()?|([\s\S])/g;
  let out = "";
  let match;
  while ((match = token.exec(sql)) !== null) {
    const [, comment, str, num, word, paren, other] = match;
    if (comment) out += `<span class="cm">${escapeHtml(comment)}</span>`;
    else if (str) out += `<span class="str">${escapeHtml(str)}</span>`;
    else if (num) out += `<span class="num">${num}</span>`;
    else if (word) {
      if (KEYWORDS.has(word.toLowerCase())) out += `<span class="kw">${escapeHtml(word)}</span>`;
      else if (paren) out += `<span class="fn">${escapeHtml(word)}</span>`;
      else out += escapeHtml(word);
      if (paren) out += escapeHtml(paren);
    } else out += escapeHtml(other);
  }
  return out;
}

function renderMarkdownLite(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// ---------------------------------------------------------------- conversation

function addMessage(kind, text, extra) {
  const empty = $("chat-empty");
  if (empty) empty.remove();
  const msg = el("div", `msg ${kind}`);
  if (kind === "bot") msg.innerHTML = renderMarkdownLite(text);
  else msg.textContent = text;
  if (extra && extra.status) {
    const status = el("span", `status ${extra.status}`, statusLabel(extra));
    msg.appendChild(document.createElement("br"));
    msg.appendChild(status);
  }
  for (const warning of (extra && extra.warnings) || []) msg.appendChild(el("span", "warn", warning));
  $("messages").appendChild(msg);
  $("messages").scrollTop = $("messages").scrollHeight;
  return msg;
}

function statusLabel(outcome) {
  const parts = [outcome.status];
  if (outcome.attempts) parts.push(`${outcome.attempts.length} SQL attempt${outcome.attempts.length === 1 ? "" : "s"}`);
  if (outcome.trace) parts.push(`${outcome.trace.llm_calls} LLM calls`, `${outcome.trace.tool_calls} tool calls`);
  if (outcome.trace && outcome.trace.latency_ms != null) parts.push(`${Math.round(outcome.trace.latency_ms)} ms`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------- steps timeline

function addStep(title, tag, tone) {
  const timeline = $("timeline");
  const placeholder = timeline.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  const li = el("li", `step ${tone || ""}`);
  const head = el("div", "head");
  head.appendChild(el("span", "title", title));
  head.appendChild(el("span", "tag", tag || ""));
  li.appendChild(head);
  timeline.appendChild(li);
  timeline.scrollTop = timeline.scrollHeight;
  return li;
}

function setTone(card, tone) {
  card.classList.remove("ok", "bad", "warn", "info");
  if (tone) card.classList.add(tone);
}

function addFeedback(card, text, tone) {
  card.appendChild(el("div", `feedback ${tone}`, text));
}

function onStep(event) {
  switch (event.kind) {
    case "link_schema": {
      const card = addStep("Schema linking", "BM25 + hashing embeddings, RRF", "info");
      const chips = el("div", "chips");
      for (const name of event.selected || []) chips.appendChild(el("span", "", name));
      for (const hit of event.value_hits || []) chips.appendChild(el("span", "dim", `${hit[0]}.${hit[1]} = '${hit[2]}'`));
      card.appendChild(chips);
      break;
    }
    case "plan":
    case "generate": {
      for (const call of event.tool_calls || []) {
        if (call.name === "run_sql") continue;
        const card = addStep(`Tool call: ${call.name}`, event.kind, "info");
        card.appendChild(el("div", "body", JSON.stringify(call.args)));
      }
      if (event.plan) {
        const card = addStep("Plan", "llm", "info");
        card.appendChild(el("div", "body", event.plan));
      }
      if (event.sql !== undefined) {
        const n = state.sqlCards.length + 1;
        const card = addStep(`SQL attempt ${n}`, "run_sql", "");
        const pre = el("pre", "sql");
        pre.innerHTML = highlightSql(event.sql);
        card.appendChild(pre);
        state.sqlCards.push(card);
      } else if (event.answer) {
        const card = addStep("Model reply (no SQL)", event.status || "", event.status === "refused" ? "bad" : "warn");
        card.appendChild(el("div", "body", event.answer));
      }
      break;
    }
    case "explore": {
      for (const tool of event.tools || []) {
        const card = addStep("Tool result", tool.tool_call_id, "");
        const preview = tool.content.split("\n").slice(0, 7).join("\n");
        card.appendChild(el("div", "body", preview + (tool.content.split("\n").length > 7 ? "\n…" : "")));
      }
      break;
    }
    case "check": {
      const card = state.sqlCards[state.sqlCards.length - 1];
      if (!card) break;
      if (event.ok) addFeedback(card, "guard + static check passed", "ok");
      else {
        setTone(card, "bad");
        addFeedback(card, event.feedback || "rejected", "bad");
      }
      break;
    }
    case "execute": {
      const card = state.sqlCards[state.sqlCards.length - 1];
      if (!card) break;
      if (event.ok) {
        setTone(card, "ok");
        addFeedback(card, `ran read-only in ${event.elapsed_ms} ms · ${event.row_count} row${event.row_count === 1 ? "" : "s"}${event.truncated ? " (truncated)" : ""}`, "ok");
      } else {
        setTone(card, "bad");
        addFeedback(card, event.error || "failed", "bad");
      }
      break;
    }
    case "verify": {
      const card = state.sqlCards[state.sqlCards.length - 1];
      if (!card) break;
      if (event.accepted) addFeedback(card, "result verified", "ok");
      else {
        setTone(card, "warn");
        addFeedback(card, event.feedback || "suspicious result", "warn");
      }
      break;
    }
    case "answer": {
      if (event.answer) {
        const card = addStep("Answer", "llm", "ok");
        card.appendChild(el("div", "body", event.answer.replace(/\*\*/g, "")));
      }
      break;
    }
    default:
      break;
  }
}

// ---------------------------------------------------------------- result, chart, trace

function renderResult(outcome) {
  const wrap = $("result");
  wrap.textContent = "";
  if (!outcome.columns.length) {
    wrap.appendChild(el("div", "placeholder", "No result for this question."));
    $("result-sub").textContent = "";
    return;
  }
  const table = el("table");
  const head = el("tr");
  for (const c of outcome.columns) head.appendChild(el("th", "", c));
  const thead = el("thead");
  thead.appendChild(head);
  table.appendChild(thead);
  const body = el("tbody");
  for (const row of outcome.rows.slice(0, 100)) {
    const tr = el("tr");
    for (const value of row) {
      const isNum = typeof value === "number";
      const shown = value === null ? "NULL" : isNum ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value);
      tr.appendChild(el("td", isNum ? "num" : "", shown));
    }
    body.appendChild(tr);
  }
  table.appendChild(body);
  wrap.appendChild(table);
  $("result-sub").textContent = `${outcome.row_count} row${outcome.row_count === 1 ? "" : "s"}${outcome.truncated ? " (truncated)" : ""}`;
}

function renderChart(spec) {
  const target = $("chart");
  target.textContent = "";
  if (!spec) {
    target.appendChild(el("div", "placeholder", "No chart for this result."));
    return;
  }
  if (typeof window.vegaEmbed !== "function") {
    target.appendChild(el("pre", "sql", JSON.stringify(spec, null, 2)));
    return;
  }
  const themed = Object.assign({}, spec, {
    width: "container",
    height: 220,
    background: "transparent",
    config: {
      axis: { labelColor: "#94a3b8", titleColor: "#94a3b8", gridColor: "#1f2a37", domainColor: "#334155", tickColor: "#334155" },
      legend: { labelColor: "#cbd5e1", titleColor: "#94a3b8" },
      title: { color: "#cbd5e1", fontSize: 12, fontWeight: 500 },
      view: { stroke: null },
      range: { category: ["#2dd4bf", "#f59e0b", "#818cf8", "#f472b6", "#a3e635", "#38bdf8"] },
      mark: { color: "#2dd4bf" },
    },
  });
  delete themed.title;
  window.vegaEmbed(target, themed, { actions: false, renderer: "svg" }).catch((err) => {
    target.textContent = `Chart failed to render: ${err}`;
  });
}

async function renderTrace(outcome) {
  const target = $("trace");
  target.textContent = "";
  const info = outcome.trace || {};
  if (!info.trace_id) return;
  let spans = [];
  try {
    const response = await fetch(`/api/trace/${outcome.thread_id}/${info.trace_id}`);
    if (response.ok) spans = await response.json();
  } catch (err) {
    target.appendChild(el("div", "placeholder", `Trace unavailable: ${err}`));
    return;
  }
  const estimated = info.tokens_estimated ? " (estimated)" : "";
  $("trace-sub").textContent = `${info.trace_id}.jsonl`;
  target.appendChild(el("div", "trace-total",
    `${spans.length} spans · ${info.llm_calls} LLM calls · ${info.tool_calls} tool calls · ` +
    `${info.tokens_in + info.tokens_out} tokens${estimated}`));
  const run = spans.find((s) => s.kind === "run");
  if (!run) return;
  const t0 = Date.parse(run.start);
  const total = Math.max(run.latency_ms, 1);
  // Spans are written when they end, so children come before parents in the file.
  // Order by start time, and put a parent before a child that started in the same ms.
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  const depthOf = (span) => {
    let d = 0;
    for (let p = span.parent_id; p && byId.has(p); p = byId.get(p).parent_id) d += 1;
    return d;
  };
  const ordered = spans
    .filter((s) => s.kind !== "run")
    .sort((a, b) => Date.parse(a.start) - Date.parse(b.start) || depthOf(a) - depthOf(b));
  for (const span of ordered) {
    const d = depthOf(span);
    const row = el("div", "span-row");
    const label = el("div", `label ${span.kind}`, span.name);
    label.style.paddingLeft = `${(d - 1) * 10}px`;
    row.appendChild(label);
    const track = el("div", "bar-track");
    const bar = el("div", `bar ${span.kind}`);
    const start = Math.max(0, Date.parse(span.start) - t0);
    bar.style.left = `${Math.min(100, (start / total) * 100)}%`;
    bar.style.width = `${Math.max(0.6, (span.latency_ms / total) * 100)}%`;
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el("div", "ms", span.kind === "event" ? "event" : `${span.latency_ms.toFixed(1)} ms`));
    const tokens = span.tokens_in + span.tokens_out;
    row.appendChild(el("div", "tok", tokens ? `${tokens} tok` : ""));
    target.appendChild(row);
  }
}

// ---------------------------------------------------------------- run a question

function resetWork() {
  state.sqlCards = [];
  $("timeline").textContent = "";
  $("steps-sub").textContent = "";
}

function finish(outcome) {
  state.running = false;
  $("ask-button").disabled = false;
  addMessage("bot", outcome.answer || "(no answer)", outcome);
  renderResult(outcome);
  renderChart(outcome.chart);
  renderTrace(outcome);
  $("steps-sub").textContent = `${outcome.steps} steps · thread ${outcome.thread_id}`;
}

function ask(question) {
  if (state.running || !question.trim()) return;
  state.running = true;
  $("ask-button").disabled = true;
  addMessage("user", question);
  resetWork();
  const url = `/api/ask/stream?q=${encodeURIComponent(question)}&thread=${state.thread}`;
  const source = new EventSource(url);
  state.source = source;
  source.addEventListener("step", (msg) => onStep(JSON.parse(msg.data)));
  source.addEventListener("done", (msg) => {
    source.close();
    finish(JSON.parse(msg.data));
  });
  source.addEventListener("error", (msg) => {
    source.close();
    state.running = false;
    $("ask-button").disabled = false;
    let text = "The connection to the server was lost.";
    if (msg.data) {
      try { text = JSON.parse(msg.data).message; } catch (_) { /* keep default */ }
    }
    addMessage("bot", text, { status: "failed" });
  });
}

async function init() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("db-pill").textContent = `${info.database} · ${info.dialect} · ${info.tables.length} tables`;
    const mode = $("mode-pill");
    mode.textContent = info.offline ? "scripted demo · no LLM" : info.model;
    if (!info.offline) mode.classList.add("online");
    $("offline-banner").hidden = !info.offline;
    for (const example of info.examples) {
      const chip = el("button", "chip", example);
      chip.type = "button";
      chip.addEventListener("click", () => { $("question").value = example; ask(example); });
      $("examples").appendChild(chip);
    }
  } catch (err) {
    $("db-pill").textContent = `server unreachable: ${err}`;
  }
  $("ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const q = $("question").value;
    $("question").value = "";
    ask(q);
  });
  $("new-chat").addEventListener("click", () => {
    state.thread = newThread();
    $("messages").textContent = "";
    resetWork();
    $("timeline").appendChild(el("li", "placeholder", "New conversation started."));
    $("result").textContent = "";
    $("chart").textContent = "";
    $("trace").textContent = "";
  });
}

document.addEventListener("DOMContentLoaded", init);
