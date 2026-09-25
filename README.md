# Local Jarvis

> **A fully local, free, and privacy-first autonomous voice assistant for Windows.**  
> Zero cloud dependencies. Zero subscriptions. Zero audio or telemetry leaving your machine.

---

## Overview

**Local Jarvis** is an open-source, hands-free personal voice assistant designed to run 100% locally on consumer Windows hardware. Powered by a local large language model, speech-to-text, neural text-to-speech, and a stateful LangGraph agent, Local Jarvis brings intelligent voice interaction, desktop automation, and long-term memory to your PC without sacrificing your privacy.

Whether you need to launch apps, control desktop windows, search the web via a private self-hosted metasearch engine, calculate complex math, set scheduled reminders, or recall past conversations, Local Jarvis executes actions instantly while keeping all processing entirely on-device.

---

## Key Features

- **Hands-Free Wake Word Detection**: Powered by `openWakeWord`, constantly listening for *"Hey Jarvis"* with low CPU overhead and near-instant response.
- **Continuous Conversation Mode**: Multi-turn hands-free dialogue with dynamic Voice Activity Detection (VAD). Speak naturally back-and-forth without repeating the wake word. Gracefully returns to standby after a configurable silence timeout or on natural dismissal phrases (*"stop"*, *"thank you jarvis"*, *"goodbye"*).
- **12 Built-in Agent Tools**:
  - 🌐 **Web Search**: Private web search via a self-hosted SearXNG Docker container.
  - 📋 **Clipboard Intelligence**: Read and summarize current clipboard contents on demand.
  - 🚀 **Universal App Launcher**: Fuzzy-matches and launches desktop apps, Start Menu entries, and Windows Store apps by name or alias.
  - 🖱️ **Windows UI Automation**: Discovers and clicks accessible UI elements (buttons, tabs, menu items) and scrolls windows using `pywinauto` (Windows UIA).
  - ⌨️ **Desktop Typing & Enter Key**: Types text directly into active windows with an interactive safety confirmation gate (`[y/N]`) to prevent unauthorized keystrokes.
  - 🧮 **Sandboxed Code Execution**: Executes mathematical expressions and calculations in an AST-validated, secure Python sandbox.
  - ⏰ **Scheduled Reminders**: SQLite-backed reminder scheduler that stores, lists, and triggers spoken voice alerts when tasks are due.
  - 📅 **System Datetime**: Dedicated real-time clock and calendar queries.
- **Persistent Long-Term Memory**: Backed by ChromaDB vector storage. Remembers user preferences, personal details, and past conversation snippets across restarts using semantic embeddings.
- **Custom 3D Wireframe Orb UI**: A borderless, always-on-top, semi-transparent glowing holographic sphere rendered in pure Python/Tkinter (multiprocessed, zero third-party 3D dependencies). Features non-focus-stealing windows, mouse drag/resize, and dynamic reactive color/size states.
- **Silent Windows Startup**: Auto-launch on Windows login via VBScript (`start_jarvis.vbs`) with no visible terminal window and background system tray controls (`pystray`).
- **Telemetry & Interaction Analytics**: Built-in SQLite logging tracking turn-by-turn latency across STT, LLM reasoning, and TTS synthesis, complete with an analysis Jupyter notebook.

---

## Architecture

Local Jarvis operates as an autonomous pipeline coordinated by an orchestrator state machine:

```
[ Microphone ]
      │
      ▼
1. Wake Word Engine (openWakeWord) ───► Listens for "Hey Jarvis" (16kHz audio stream)
      │
      ▼
2. Speech-to-Text (faster-whisper) ───► Transcribes recorded user speech to text
      │
      ▼
3. Agent Brain (LangGraph + Ollama) ──► Evaluates user intent with llama3.2:3b
      │   ├─ ChromaDB Memory (Vector context retrieval & storage)
      │   └─ 12 Built-in Tools (Web, Desktop, Reminders, Sandbox, etc.)
      │
      ▼
4. Text-to-Speech (Piper TTS) ────────► Synthesizes low-latency neural voice response
      │
      ▼
[ Speaker & Dynamic Orb/Tray Feedback ]
```

- **Wake Word**: `openWakeWord` processes continuous audio chunks using an ONNX runtime model. When the wake threshold is met, it captures follow-up speech until silence is detected.
- **STT**: `faster-whisper` (CTranslate2-optimized Whisper `small` model) converts speech to text on CPU/GPU with high accuracy and low latency.
- **Brain & Tools**: A stateful `LangGraph` pipeline binds to an `Ollama` instance running `llama3.2:3b`. The agent assesses whether tools are needed, checks semantic memory in `ChromaDB`, executes tools deterministically, and crafts natural spoken answers.
- **TTS**: `piper-tts` generates clean, natural speech output using trained ONNX voice checkpoints (`ryan` or `lessac`) streamed directly to audio output devices.

---

## Tech Stack

| Component | Technology | Role / Description |
| :--- | :--- | :--- |
| **Wake Word** | `openWakeWord` (ONNX) | Lightweight, real-time wake word detection for *"Hey Jarvis"* |
| **Speech-to-Text (STT)** | `faster-whisper` | Fast CTranslate2 implementation of OpenAI Whisper (`small` model) |
| **LLM Engine** | `Ollama` (`llama3.2:3b`) | Local GPU-accelerated language model reasoning |
| **Agent Framework** | `LangGraph` / `LangChain` | Stateful conversation graph, conditional tool routing, and memory injection |
| **Text-to-Speech (TTS)** | `piper-tts` | Fast, high-quality neural voice synthesis on CPU/GPU |
| **Web Search** | `SearXNG` (Docker) | Self-hosted, private metasearch engine (no tracking, no API keys) |
| **Long-Term Memory** | `ChromaDB` | Embedded vector database for semantic conversation recall |
| **Desktop Automation** | `pywinauto` & `pyautogui` | Windows Accessibility (UIA) clicks, window focus, typing, and scrolling |
| **Data Storage** | `SQLite` | Persistent storage for scheduled reminders and performance telemetry |
| **User Interface** | `Tkinter` (Custom 3D) & `pystray` | Borderless 3D wireframe orb overlay & taskbar notification tray icon |

---

## Getting Started

### Prerequisites

1. **Operating System**: Windows 10 or 11 (64-bit).
2. **Python**: Version **3.13** (or 3.10+) installed with `PATH` configured.
3. **Docker Desktop**: Required to host the private SearXNG search engine container.
4. **Ollama**: Download and install [Ollama for Windows](https://ollama.com/) (runs with native NVIDIA GPU acceleration).

---

### Installation & Setup

1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/AnkurGaur777/voice-assistant.git
   cd voice-assistant
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Python Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Start the SearXNG Search Service**:
   ```powershell
   docker compose up -d
   ```
   *Verify it is running by visiting `http://localhost:8080` in your browser.*

5. **Pull the Local LLM**:
   ```powershell
   ollama pull llama3.2:3b
   ```

6. **Run Local Jarvis**:
   ```powershell
   python src/orchestrator.py
   ```

---

## Silent Background Startup (`shell:startup`)

To run Local Jarvis automatically in the background every time you log in to Windows:

1. Press <kbd>Win</kbd> + <kbd>R</kbd>, type `shell:startup`, and press <kbd>Enter</kbd>.
2. In File Explorer, navigate to your `voice-assistant` directory.
3. Right-click `start_jarvis.vbs` -> **Show more options** -> **Create shortcut**.
4. Cut and paste the newly created shortcut into the `Startup` folder.

> [!NOTE]
> `start_jarvis.vbs` runs silently without displaying any command prompt window. You can monitor or exit Jarvis at any time via the taskbar notification tray icon or right-clicking the floating 3D orb.

---

## Usage & Example Commands

Say **"Hey Jarvis"** to wake the assistant, then ask your question or give an instruction:

### Desktop & App Control
- *"Hey Jarvis, open Notepad."*
- *"Open WhatsApp and click Chats."*
- *"Scroll down."*
- *"Type 'Meeting confirmed for 3 PM' and press enter."* *(Prompts for safety confirmation before sending)*

### Web Search & Information
- *"Hey Jarvis, search the web for the latest Mars rover discoveries."*
- *"What is the current time and today's date?"*
- *"Summarize what's on my clipboard."*

### Sandboxed Math & Logic
- *"What is 358% of 340?"*
- *"Calculate 45 times 18 plus 120."*

### Scheduled Reminders
- *"Remind me to call the doctor tomorrow at 10 AM."*
- *"What are my pending reminders?"*

### Long-Term Memory Recall
- *"My favorite programming language is Python."*
- *(Later in the session or days later)*: *"Hey Jarvis, what's my favorite programming language?"*

### Hands-Free Continuous Conversation
After waking Jarvis once, you can have a natural multi-turn conversation without repeating the wake word:
```
User:      "Hey Jarvis, what is quantum computing?"
Jarvis:    "Quantum computing is a field of computing based on the principles of quantum mechanics..."
User:      "How does that differ from classical computers?"
Jarvis:    "Classical computers process information in binary bits (0 or 1), while quantum computers use qubits..."
User:      "Thank you Jarvis, that's all."
Jarvis:    [Reverts to standby listening mode]
```

---

## CLI Flags Reference

You can customize runtime behavior using command-line arguments:

```powershell
python src/orchestrator.py [OPTIONS]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `llama3.2:3b` | Ollama model name to use for reasoning |
| `--base-url` | `str` | `http://localhost:11434` | Ollama server API endpoint |
| `--whisper-model` | `str` | `small` | faster-whisper model size (`tiny`, `base`, `small`) |
| `--voice` | `str` | `ryan` | Piper TTS neural voice model (`ryan`, `lessac`) |
| `--threshold` | `float` | `0.35` | openWakeWord detection confidence threshold (0.0 to 1.0) |
| `--device` | `int` | `None` | Audio input device index (defaults to system default mic) |
| `--conversation-timeout` | `float` | `60.0` | Inactivity silence seconds in continuous mode before standby |
| `--no-continuous` | `flag` | `False` | Disable continuous mode (requires wake word on every turn) |
| `--stop-phrases` | `str` | `None` | Comma-separated list of custom conversation stop phrases |
| `--no-orb` | `flag` | `False` | Disable the floating 3D glowing wireframe orb UI |
| `--no-tray` | `flag` | `False` | Run in console-only mode without system tray icon |
| `--no-memory` | `flag` | `False` | Disable ChromaDB persistent vector memory |
| `--no-logging` | `flag` | `False` | Disable interaction logging to SQLite |
| `--no-greeting` | `flag` | `False` | Disable spoken greeting on launch |
| `--greeting-text` | `str` | `"Hello! I'm ready..."` | Custom text for spoken startup greeting |
| `--no-wake-ack` | `flag` | `False` | Disable the audible acknowledgment (*"Yes?"*) on wake word |
| `--barge-in` | `flag` | `False` | Enable experimental speech interruption during TTS playback |
| `--barge-in-threshold` | `float` | `400.0` | Mic RMS energy threshold for barge-in interruption |
| `--list-launchable-apps` | `flag` | `False` | Scan and print all discoverable Windows applications, then exit |
| `--list-ui-elements APP` | `str` | `None` | Inspect accessible UI elements for a target app, then exit |
| `--debug` | `flag` | `False` | Enable verbose wake word and audio diagnostics |

---

## Known Limitations

- **Electron & Web App UI Accessibility**: Applications rendered entirely via Chromium/Electron or custom hardware-accelerated canvas surfaces without native Windows UI Automation (UIA) hooks may not expose individual buttons or tabs to `pywinauto`.
- **Barge-In Interruption (Experimental)**: Speech interruption during active text-to-speech playback is disabled by default. Depending on microphone sensitivity and speaker volume, acoustic echo can cause false triggers or missed detections.
- **Small (3B) Parameter Model Quirks**: Local Jarvis defaults to `llama3.2:3b` for fast on-device inference on consumer GPUs. While highly capable, small models can occasionally attempt plain-text arithmetic on ambiguously phrased queries or require precise phrasing for complex multi-tool workflows.
- **Language Support**: Designed and configured out-of-the-box for **English** voice interaction.

---

## Data Analysis & Benchmarking

Local Jarvis logs every interaction into a local SQLite database (`analysis/interactions.db`), recording timestamped metrics including:
- Speech-to-Text transcription duration (`stt_seconds`)
- LangGraph reasoning and tool execution latency (`brain_seconds`)
- Text-to-Speech audio generation latency (`tts_seconds`)
- Tools invoked per turn and task success rates

Check out [`analysis/interaction_analysis.ipynb`](analysis/interaction_analysis.ipynb) to inspect performance distributions, latency percentiles across pipeline stages, tool usage heatmaps, and STT benchmark evaluations.

---

## Development & AI-Assisted Design

Local Jarvis was architected and built iteratively using modern AI-assisted engineering practices:
- **Google Antigravity**: Autonomous pair-programming agent environment for rapid implementation and refactoring.
- **GSD (Get Stuff Done)**: Spec-driven hierarchical planning and execution framework.
- **CodeRabbit**: Automated PR and static code review ensuring safe tool execution, robust error handling, and lint hygiene.

---

## License

This project is open-source and released under the [MIT License](LICENSE).
