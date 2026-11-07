// Minimal UI for the RAG Engine API (same-origin: /health, /ingest, /ask).

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");
const docEl = document.getElementById("doc");
const srcEl = document.getElementById("src");
const ingestBtn = document.getElementById("ingest");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addCitations(container, sources) {
  if (!sources || !sources.length) return;
  const d = document.createElement("details");
  d.className = "cites";
  const s = document.createElement("summary");
  s.textContent = `${sources.length} source(s)`;
  d.appendChild(s);
  const ul = document.createElement("ul");
  for (const src of sources) {
    const li = document.createElement("li");
    const score = src.score !== undefined ? ` (score ${src.score})` : "";
    li.textContent = `${src.source}${score}: ${src.excerpt || ""}`;
    ul.appendChild(li);
  }
  d.appendChild(ul);
  container.appendChild(d);
}

async function refreshHealth() {
  try {
    const h = await (await fetch("/health")).json();
    document.getElementById("backend").textContent = `backend: ${h.backend}`;
    document.getElementById("chunks").textContent = `index: ${h.indexed_chunks} chunks`;
  } catch { /* ignore */ }
}

ingestBtn.addEventListener("click", async () => {
  const text = docEl.value.trim();
  if (!text) return;
  ingestBtn.disabled = true;
  try {
    const res = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source: srcEl.value.trim() || "pasted.txt" }),
    });
    const data = await res.json();
    addMessage(`Ingested ${data.chunks_added} chunk(s). Index now ${data.total_chunks}.`, "bot");
    docEl.value = "";
    refreshHealth();
  } catch (err) {
    addMessage(`Ingest error: ${err.message}`, "bot");
  } finally {
    ingestBtn.disabled = false;
  }
});

async function ask(question) {
  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  const bot = addMessage("…", "bot");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    bot.textContent = data.answer || "(no answer)";
    addCitations(bot, data.sources);
  } catch (err) {
    bot.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) ask(q);
});

// Example question buttons: click to ask immediately.
for (const btn of document.querySelectorAll("#examples button")) {
  btn.addEventListener("click", () => ask(btn.textContent.trim()));
}

refreshHealth();
