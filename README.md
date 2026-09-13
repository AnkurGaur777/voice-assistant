# Local Jarvis - Autonomous Hands-Free Voice Assistant

An autonomous, fully local voice assistant running on Windows with wake word detection, speech-to-text, LLM agent reasoning with tool invocation and persistent memory, and neural text-to-speech synthesis.

---

## Key Features

1. **Wake Word Detection**: Hands-free detection for *"Hey Jarvis"* using openWakeWord (ONNX).
2. **Continuous Conversation Mode**:
   - Multi-turn hands-free dialogue without repeating *"Hey Jarvis"*.
   - Dynamic VAD speech segmentation with a 60.0-second silence timeout safety net (quietly reverts to standby while preserving conversation context).
   - Configurable stop phrase recognition (*"stop"*, *"goodbye"*, *"that's all"*, *"thank you jarvis"*).
3. **Local LLM & LangGraph Agent**:
   - Ollama-powered LLM reasoning (`llama3.2:3b` on RTX GPU).
   - 10 Built-in Tools: System Datetime, SearXNG Web Search, Clipboard Read/Summarize, Desktop App Launcher, Desktop Keystroke Injection & Enter/Send, Python Sandboxed Math, and SQLite Reminders.
   - Long-term memory backed by ChromaDB vector store.
4. **Desktop Automation & Type-and-Send**:
   - `type_text(text, press_enter=...)`: Types text into active applications (WhatsApp, Messages, Notepad, email compose) with combined *"Type and press enter?"* confirmation.
   - `press_enter_key()`: Standalone tool to send/submit in the focused window.
   - Mandatory interactive `[y/N]` confirmation gate prevents unauthorized keystrokes.
5. **System Tray Integration**:
   - Dynamic tray icon via `pystray` reflecting real-time pipeline status:
     - 🔵 **Listening / Idle** (Cyan with microphone)
     - 🟣 **Active Conversation** (Vibrant violet with speech bubble)
     - 🟠 **Processing** (Amber with activity dots)
     - 🟢 **Speaking** (Emerald green with speaker and sound waves)
     - 🔴 **Error** (Crimson with exclamation point)
   - Right-click menu -> *"Quit Jarvis"* for graceful shutdown.

---

## Windows Automatic Startup (`shell:startup`)

To have Local Jarvis start automatically in the background when you log in to Windows:

1. Press <kbd>Win</kbd> + <kbd>R</kbd> to open the Windows **Run** dialog.
2. Type `shell:startup` and press <kbd>Enter</kbd>. This opens your personal Startup folder:
   ```
   C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
   ```
3. In File Explorer, navigate to the `voice-assistant` project directory.
4. Right-click [`start_jarvis.vbs`](start_jarvis.vbs) -> select **Show more options** -> **Create shortcut**.
5. Move (or copy/paste) the newly created shortcut into the `Startup` folder.

**How it works:**
- On login, Windows executes `start_jarvis.vbs`.
- It activates the Python virtual environment and launches `src/orchestrator.py` **silently with no visible console window**.
- The Jarvis system tray icon will appear in the taskbar notification tray.
- Diagnostic logs from the startup process are saved to `jarvis_startup.log`.

---

## Manual Running

### Foreground Console Mode
```powershell
# Activate venv and run
.\.venv\Scripts\activate
python src\orchestrator.py
```

### Silent Background Launch
Double-click `start_jarvis.vbs` or run:
```powershell
wscript.exe start_jarvis.vbs
```

### CLI Options
```
--model MODEL             Ollama model name (default: llama3.2:3b)
--whisper-model SIZE      faster-whisper size (tiny, base, small; default: small)
--voice VOICE             Piper TTS voice (ryan, lessac; default: ryan)
--threshold FLOAT         Wake word confidence threshold (default: 0.35)
--conversation-timeout S  Silence timeout for continuous conversation (default: 60.0)
--no-continuous           Disable multi-turn hands-free continuous conversation
--stop-phrases LIST       Comma-separated custom stop phrases
--no-tray                 Run in console mode without tray icon
--no-memory               Disable ChromaDB memory
--debug                   Enable verbose wake word and audio logs
```
