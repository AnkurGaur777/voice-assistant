# Local Jarvis — Execution Guide

Copy-paste-ready steps for Antigravity + GSD + CodeRabbit. Do these **in order** — each phase depends on the last one working.

**Your hardware:** Ryzen 5, 16GB DDR5, RTX 3050 6GB → Llama 3.1 8B Q4 on GPU, faster-whisper on CPU, Piper/Kokoro TTS on CPU.

**Total estimated time:** ~4–6 weeks at 2–3 hrs/day (part-time/evenings), or ~2 weeks if full-time. Time estimates below assume part-time, solo, first time building this kind of pipeline.

---

## Phase 0 — Repo & Environment Setup
**Time: 3–4 hours**

1. Create GitHub repo `local-jarvis`, clone it, open in Antigravity.
2. Run in terminal:
   ```bash
   npx get-shit-done-cc --antigravity --local
   ```
3. In Antigravity, type:
   ```
   /discuss
   Set up the project skeleton for a local voice assistant.
   Need: Python venv, requirements.txt, docker-compose.yml with services
   for ollama and searxng, .gitignore (exclude venv, models, .env, *.wav),
   and the folder structure: src/, src/agent/tools/, analysis/, tray/, .planning/
   ```
4. `/plan` → review the plan file → `/execute` → `/verify`.
5. `git push`, open a PR, let CodeRabbit review it, merge.

---

## Phase 1 — Wake Word + Audio Capture
**Time: 4–6 hours**

```
/discuss
Implement wake-word detection using openWakeWord and audio capture using
sounddevice. On wake word detection, start recording to a ring buffer,
stop on ~1s of silence (simple VAD), save the clip as a temp .wav file.
Output: src/wake_word.py, runnable standalone, prints "Wake word detected"
and saves the clip path.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

**Test yourself:** say your wake word out loud, confirm a `.wav` file appears.

---

## Phase 2 — Speech-to-Text (+ benchmark on your hardware)
**Time: 6–8 hours** (includes benchmarking)

```
/discuss
Add faster-whisper transcription in src/stt.py, running on CPU (device="cpu").
Take the .wav path from Phase 1 and return transcribed text. Also add a
small benchmark script (analysis/stt_benchmark.py) that times tiny/base/small
models on a sample clip and logs latency + output to a CSV for comparison.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

**Do this yourself after:** run the benchmark script, pick the model (base or small is usually the sweet spot on a Ryzen 5) and hardcode that choice into `stt.py`.

---

## Phase 3 — The Brain: Ollama + LangGraph Skeleton
**Time: 1.5–2 days**

1. Pull the model locally first (not through the agent):
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```
2. In Antigravity:
   ```
   /discuss
   Build a minimal LangGraph agent in src/agent/graph.py that connects to
   local Ollama (model: llama3.1:8b-instruct-q4_K_M). Simple state machine:
   input -> llm_node -> output, text in / text out, no tools yet. Include
   a small CLI test harness (src/agent/test_cli.py) so I can chat with it
   from the terminal.
   ```
3. `/plan → /execute → /verify` → PR → CodeRabbit → merge.

**Test yourself:** chat with it via the CLI, confirm response latency (this is your GPU baseline).

---

## Phase 4 — Jarvis Powers: Tools (the big one)
**Time: 4–6 days** — do this as 5 separate mini-cycles, not one giant one. GSD works best atomic; don't let the agent write all 5 tools in one `/execute`.

**4a. Web search (1 day)**
```
/discuss
Add a web_search tool in src/agent/tools/web_search.py using a self-hosted
SearXNG instance (add searxng to docker-compose.yml if not already there).
Register it as a LangGraph tool the agent can call. Test: agent should be
able to answer "what's today's date" or a live-info question by searching.
```

**4b. Clipboard (half day)**
```
/discuss
Add read_clipboard and summarize_clipboard tools using pyperclip, in
src/agent/tools/clipboard.py. Register with the LangGraph agent.
```

**4c. App control (1 day)**
```
/discuss
Add open_application and type_text tools using PyAutoGUI in
src/agent/tools/desktop.py. Include basic safety: confirm before typing
into an unknown active window.
```

**4d. Code execution (1 day) — be careful here**
```
/discuss
Add a run_python tool that executes short Python snippets in a SANDBOXED
subprocess with a timeout (no filesystem/network access), for quick
calculations. src/agent/tools/sandbox.py.
```
⚠️ **Flag this PR for extra CodeRabbit scrutiny** — this is the highest-risk code in the whole project (arbitrary code execution). Read CodeRabbit's comments carefully before merging.

**4e. Reminders (1 day)**
```
/discuss
Add a set_reminder tool backed by SQLite (src/agent/tools/reminders.py)
plus a background scheduler thread that checks due reminders and prints/
triggers a callback. Register as a LangGraph tool.
```

Each sub-phase: `/plan → /execute → /verify` → PR → CodeRabbit → merge, before starting the next.

---

## Phase 5 — Text-to-Speech
**Time: 1 day**

```
/discuss
Add TTS output using Piper (CPU-based) in src/tts.py. Take agent's text
response and speak it. Stream playback so speech starts as soon as the
first sentence/chunk of the LLM response is ready, not after the full
response finishes generating.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

---

## Phase 6 — Memory (Chroma)
**Time: 1.5–2 days**

```
/discuss
Add local vector memory using ChromaDB in src/agent/memory.py. Before
each LLM call, embed the current query and retrieve top-3 relevant past
exchanges to inject as context. After each exchange, store it. Persist
Chroma data to disk so memory survives restarts.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

---

## Phase 7 — Full Pipeline + Tray UI
**Time: 2–3 days**

```
/discuss
Build src/orchestrator.py that wires everything together: wake word ->
STT -> LangGraph agent (with all tools + memory) -> TTS, as one continuous
loop. Add a system tray icon (pystray) with 4 states: idle, listening,
processing, speaking, that updates as the pipeline progresses.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

**Test yourself:** run the full loop end-to-end, time it from wake word to spoken response start. This is your real latency number.

---

## Phase 8 — Evaluation & Data Layer
**Time: 1.5–2 days**

```
/discuss
Add interaction logging to SQLite: timestamp, transcript, tools_called,
latency_ms per stage (STT/LLM/TTS), success/failure. Log every full
pipeline run automatically from orchestrator.py.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

**Then, yourself (not the agent — this is your DS/DE portfolio piece):**
- Build `analysis/analysis.ipynb` with pandas + seaborn: latency breakdown by stage, tool-call frequency, response time trend over your test runs.
- This notebook is what makes the repo stand out — do this part hands-on rather than delegating it.

---

## Phase 9 — Packaging & GitHub Release
**Time: 1 day**

```
/discuss
Finalize docker-compose.yml so `docker compose up` starts ollama + searxng
cleanly. Write a README.md with architecture overview, setup instructions,
and a placeholder for a demo GIF.
```
`/plan → /execute → /verify` → PR → CodeRabbit → merge.

Then manually:
```bash
git tag v1.0
git push origin v1.0
```
Record a short demo clip/GIF, add it to the README.

---

## Time Summary

| Phase | Time |
|---|---|
| 0 — Setup | 3–4 hrs |
| 1 — Wake word | 4–6 hrs |
| 2 — STT | 6–8 hrs |
| 3 — Brain skeleton | 1.5–2 days |
| 4 — Tools (5 sub-phases) | 4–6 days |
| 5 — TTS | 1 day |
| 6 — Memory | 1.5–2 days |
| 7 — Full pipeline + UI | 2–3 days |
| 8 — Evaluation | 1.5–2 days |
| 9 — Packaging | 1 day |
| **Total** | **~18–25 working days** → **4–6 weeks part-time** |

## Rules of thumb while you go
- Never let GSD skip `/plan` review — that's your checkpoint to catch scope creep before code gets written.
- Always merge through a PR, never direct-push agent code, so CodeRabbit gets a look.
- Test each phase yourself before starting the next — don't stack unverified phases.
- Phase 4d (code execution) and anything touching subprocess/filesystem is your highest-risk code — slow down there.
