# Local Jarvis — Roadmap

A fully local, Jarvis-style voice assistant: wake word → listen → understand → **search / act / answer** → speak. Zero cloud, zero subscription, 100% on your machine.

This builds directly on your uploaded spec (`Local_VoiceOS_Project_Plan.md`), upgraded from "voice-to-action" to an actual **agent with tools** (web search, file ops, app control, code execution) — the Jarvis part.

---

## 0. What Changes From Your Original Spec

Your original plan is a solid pipeline (wake word → STT → LLM → TTS + desktop automation). The Jarvis upgrade adds one thing: **a tool-using agent brain instead of a single-shot LLM call.** Concretely:

| Original | Upgraded |
|---|---|
| LLM answers from its own knowledge | LLM can call a **search tool** (local SearXNG or DuckDuckGo) to get live info |
| Fixed 3 tools (`type_text`, `open_app`, `summarize_clipboard`) | Growing **tool registry**: web search, file search, calendar/reminders, code execution, app launching, screen/clipboard reading |
| No memory across sessions | **Local vector memory** (Chroma) so it remembers past conversations/preferences |
| One-shot response | **Multi-step planning** via LangGraph — it can chain 2–3 tool calls before answering ("search flight prices, then draft an email") |

---

## 1. Final Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Wake word | `openWakeWord` | lightweight, always-on, free |
| Audio I/O | `sounddevice` / `PyAudio` | mic capture, playback |
| STT | `faster-whisper` (small/medium model) | fast, accurate, local |
| LLM runtime | `Ollama` running `Llama 3.1 8B` or `Phi-4` | free local inference |
| Agent orchestration | `LangGraph` (+ `LangChain` tool wrappers) | stateful, multi-step tool use |
| Model gateway (optional) | `LiteLLM` | swap models without touching agent code |
| **Search tool** | `SearXNG` (self-hosted, Docker) or DuckDuckGo HTML API | keeps "fully local/free" promise — SearXNG is a self-hosted metasearch engine, no API key |
| **Memory** | `ChromaDB` (local vector store) | remembers past turns/preferences |
| **Task tools** | `PyAutoGUI`, `subprocess`, OS Accessibility APIs, `pyperclip` | app control, clipboard, keystrokes |
| TTS | `Kokoro TTS` or `Piper` | natural local speech |
| Containerization | `Docker` / `docker-compose` | isolate Ollama, SearXNG, TTS as services |
| UI | Python `pystray` (tray icon) or a minimal `Tkinter`/`Textual` overlay | listening/processing/acting indicator |
| Data/eval layer (your DS skills) | `pandas`, `matplotlib`/`seaborn`, `SQLite` | log every interaction (latency, transcript accuracy, tool calls) for analysis — this is what makes the GitHub repo look like a *data scientist's* project, not just a hobby script |

---

## 2. Phase-by-Phase Roadmap

### Phase 0 — Repo & Environment Setup
- Create GitHub repo, clone locally, open in Antigravity.
- `docker-compose.yml` scaffolding for: `ollama`, `searxng`, (later) `chroma`.
- Python venv, `requirements.txt`, `.gitignore` (exclude models, `.env`, audio caches).
- Install GSD and set up your first PRD (see Section 3).

### Phase 1 — Audio Capture + Wake Word
- Integrate `openWakeWord`, custom trigger phrase.
- Build a simple ring-buffer recorder that activates on wake word, stops on silence (VAD).
- **Deliverable:** script that prints "Wake word detected" and saves a `.wav` clip.

### Phase 2 — Speech-to-Text
- Wire `faster-whisper` to transcribe the captured clip.
- Benchmark model sizes (tiny/base/small) for your CPU/GPU — log latency + WER to a CSV. This is a natural place to apply your pandas/matplotlib skills (latency-vs-accuracy chart).

### Phase 3 — The Brain: Ollama + LangGraph
- Stand up Ollama in Docker, pull Llama 3.1 8B (or Phi-4 if RAM-constrained).
- Build a minimal LangGraph state machine: `input → llm_node → (tool_call? → tool_node → llm_node) → output`.
- **Deliverable:** text-in/text-out agent that can decide when to call a tool vs. answer directly.

### Phase 4 — Give It Jarvis Powers (Tools)
This is the core upgrade. Implement as LangGraph tools, one at a time:
1. `web_search(query)` → SearXNG or DuckDuckGo, returns top results.
2. `read_clipboard()` / `summarize_clipboard()`.
3. `open_application(name)`, `type_text(text)` via PyAutoGUI.
4. `run_python(code)` — sandboxed code execution for quick calculations (careful: sandbox it, don't give raw shell access).
5. `set_reminder(text, time)` → local SQLite-backed reminder store + a background scheduler.
- **Deliverable:** "What's the weather in Delhi and remind me to check again in an hour" → chains search + reminder tool.

### Phase 5 — Text-to-Speech
- Wire Kokoro/Piper for spoken output.
- Add streaming so it starts speaking before the full LLM response finishes generating (biggest lever for your <2s latency goal).

### Phase 6 — Memory
- Add ChromaDB: embed and store each exchange; retrieve relevant past context before each LLM call.
- Gives it the "knows your preferences" Jarvis feel.

### Phase 7 — UI/Tray + Full Pipeline Stitching
- Tray icon with 3 states: idle / listening / processing / speaking.
- End-to-end: wake word → STT → agent (with tools) → TTS, all through one orchestrator script.

### Phase 8 — Evaluation & Data Layer (your differentiator)
- Log every interaction to SQLite: `timestamp, transcript, intent, tools_called, latency_ms, success`.
- Build a small notebook (`analysis.ipynb`) with pandas/seaborn: latency breakdown by pipeline stage, tool-call frequency, STT accuracy over time.
- This turns "I built a voice assistant" into "I built and *measured* a voice assistant" — much stronger for a DS/DE portfolio.

### Phase 9 — Packaging & GitHub
- Finalize `docker-compose.yml` so a stranger can `docker compose up` + `pip install -r requirements.txt` and run it.
- Write README with architecture diagram, setup steps, demo GIF.
- Tag a `v1.0` release.

---

## 3. Working With Antigravity + GSD + CodeRabbit

You said you'll use **Antigravity** (Google's agentic IDE) with the **GSD** ("Get Shit Done") and **CodeRabbit** plugins. Here's how to actually run this project through them.

### 3.1 Install GSD into Antigravity
GSD is a spec-driven workflow layer (structured `PLAN → EXECUTE → VERIFY` cycle) that stops the agent from wandering off-spec. Install it as an Antigravity skill:
```bash
npx get-shit-done-cc --antigravity --local
```
This drops workflow files into your project (`.agent/workflows/`, `.agent/skills/`).

### 3.2 The GSD Cycle — run this per Phase above
For **each phase** in Section 2, treat it as one GSD cycle:
1. `/discuss` — tell the agent what the phase needs to deliver (paste the phase's bullet points).
2. `/plan` — GSD generates an atomic, file-level plan (`PLAN.md`) before any code is written. **Read this before approving** — this is where you catch scope creep.
3. `/execute` — Antigravity's agent (Gemini 3 or Claude Sonnet, your choice in Antigravity's model picker) writes the code in atomic commits.
4. `/verify` — GSD's verifier checks the deliverable actually matches the plan (not just "tests pass").

Do **not** jump straight to Phase 4 (tools/search) — GSD works best when each phase is small and verified before the next starts, and it maps cleanly onto your phase list above.

### 3.3 Antigravity Agent Manager settings
- Use **Agent-assisted mode** (not full Autopilot) while you're still learning the codebase — you approve each step.
- Switch to **Autopilot** once you trust the pattern (e.g., repetitive tool-wiring in Phase 4).
- Set **Terminal Policy** to "Agent Decides" so it doesn't ask permission for routine commands like `pip install`.

### 3.4 CodeRabbit — your reviewer
CodeRabbit reviews every commit/PR the agent generates, catching things like unhandled exceptions in the audio loop, credential leaks in Docker configs, or inefficient embedding calls.
- Connect the CodeRabbit GitHub App to your repo (not just local — it works best on PRs).
- Workflow: GSD's `/execute` commits to a feature branch → open a PR → CodeRabbit auto-reviews → fix flagged issues → merge.
- Treat CodeRabbit comments as a second opinion before you trust agent-written code that touches the filesystem or subprocess calls (Phase 4 tools) — that's the highest-risk code in this project.

### 3.5 Suggested repo structure for GSD + GitHub
```
local-jarvis/
├── .agent/                # GSD workflows/skills
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── wake_word.py
│   ├── stt.py
│   ├── agent/
│   │   ├── graph.py       # LangGraph state machine
│   │   └── tools/         # web_search.py, clipboard.py, reminders.py, ...
│   ├── tts.py
│   └── orchestrator.py    # ties the whole pipeline together
├── analysis/
│   └── analysis.ipynb     # Phase 8 evaluation notebook
├── tray/
│   └── tray_app.py
├── .planning/              # GSD's PLAN.md / SUMMARY.md artifacts per phase
└── README.md
```

---

## 4. Order of Operations (TL;DR checklist)

- [ ] Phase 0: repo, Docker, venv, GSD install
- [ ] Phase 1: wake word + recording
- [ ] Phase 2: STT working end-to-end
- [ ] Phase 3: Ollama + LangGraph skeleton agent
- [ ] Phase 4: search + task tools (the Jarvis part)
- [ ] Phase 5: TTS output
- [ ] Phase 6: memory (Chroma)
- [ ] Phase 7: full pipeline + tray UI
- [ ] Phase 8: logging + pandas/seaborn evaluation notebook
- [ ] Phase 9: README, docker-compose polish, push `v1.0` to GitHub

Run each checked phase through the GSD `/discuss → /plan → /execute → /verify` cycle in Antigravity, with CodeRabbit reviewing the resulting PR before merge.
