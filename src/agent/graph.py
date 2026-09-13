"""
Local Jarvis - LangGraph Brain & LLM Orchestration Module (Phase 3)

This module builds the core LangGraph state machine connecting to a local Ollama
instance (model: llama3.1:8b, base_url: http://localhost:11434) running on GPU.

Architecture:
  START -> llm_node -> END
  (Designed as a modular, extensible foundation ready for tool_node integration in Phase 4)
"""

import json
import os
import re
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.tools.clipboard import read_clipboard, summarize_clipboard
from src.agent.tools.datetime_tool import get_current_datetime
from src.agent.tools.desktop import open_application, press_enter_key, type_text
from src.agent.tools.reminders import list_reminders, set_reminder
from src.agent.tools.sandbox import run_python
from src.agent.tools.web_search import web_search
from src.agent.memory import (
    format_context_for_prompt,
    retrieve_relevant_context,
    store_exchange,
)
from src.tts import sanitize_speech_text

# --- Default Configuration ---
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TOOLS = [
    get_current_datetime,
    web_search,
    read_clipboard,
    summarize_clipboard,
    open_application,
    type_text,
    press_enter_key,
    run_python,
    set_reminder,
    list_reminders,
]

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, capable, and intelligent local voice assistant.\n\n"
    "FEW-SHOT CONVERSATIONAL EXAMPLES (NO TOOLS):\n"
    "User: \"my favorite color is blue\"\n"
    "Assistant: \"Got it, I'll remember that!\" (plain conversational acknowledgment, NO tool call)\n\n"
    "User: \"Hello world.\"\n"
    "Assistant: \"Hello! How can I help you today?\" (conversational greeting, NO tool called - NEVER claim typing or desktop actions occurred or were cancelled)\n\n"
    "CRITICAL CONVERSATIONAL & TOOL CALLING RULES:\n"
    "- NEVER FABRICATE ACTION OUTCOMES WHEN NO TOOL WAS CALLED: If NO tool was called in the current turn, you must NEVER claim or imply that any action-related outcome occurred (such as typing, opening an application, cancelling an action, rejecting keystrokes, sending text, or executing a command). Action-related phrases like 'Typing was cancelled into...', 'Opening...', 'Typed text into...', 'Sent message to...', or 'Action cancelled' must ONLY EVER appear when a corresponding live tool result (ToolMessage) actually exists in the message history for this current turn. If the user's utterance is a statement, greeting, or unclear remark (e.g. 'Hello world.') where no tool was called, respond strictly as a normal conversational turn or ask for clarification — NEVER fabricate, hallucinate, or assume that an action was attempted, executed, or cancelled.\n"
    "- APPLICATION & DESKTOP ACTIONS: ALWAYS invoke the `open_application` tool whenever the user asks to open, launch, or start an application "
    "(e.g. \"open notepad\" -> invoke `open_application(app_name=\"notepad\")`, \"launch chrome\" -> invoke `open_application(app_name=\"chrome\")`). "
    "NEVER output plain text saying \"Opening Notepad.\" or \"I'll open...\" without invoking `open_application`! You cannot open applications without calling the tool.\n"
    "- DESKTOP TYPING & ENTER KEY: ALWAYS invoke `type_text(text=\"...\", press_enter=...)` when asked to type text into the active window, "
    "and `press_enter_key()` when asked to press enter or send. NEVER simulate or narrate typing in plain text.\n"
    "- DATE & TIME QUESTIONS: ALWAYS call the `get_current_datetime` tool for any questions asking for the current date, today's date, current time, day of the week, month, or year (e.g. \"what's the date of today\", \"what time is it\", \"what day is today\"). You do not have an internal clock, so you MUST query `get_current_datetime` for real-time date and time. NEVER state that the date or time is not available or a dynamic value. When synthesizing the final response from `get_current_datetime`, ALWAYS speak a single, concise natural sentence (e.g. \"It's Sunday, September 13th, 1:46 PM\" or \"The time is 1:46 PM\"). NEVER read bullet points, field labels, or raw tool output verbatim.\n"
    "- MATHEMATICAL CALCULATIONS & PERCENTAGES: ALWAYS use the `run_python` tool to evaluate math, arithmetic, and percentages (e.g. \"what is 358% of 340\" -> invoke `run_python` with code '340 * 3.58'). NEVER call `web_search` for math, arithmetic, or percentage questions.\n"
    "- When the user makes a statement sharing personal information/preferences (not a question, not a request to DO something), "
    "respond conversationally acknowledging it (e.g. \"Got it, I'll remember that!\") and do NOT call any tool. "
    "Tools should only be called when the user is explicitly asking to perform an action (set a reminder, search, calculate, etc.), "
    "not when they're just sharing a fact.\n"
    "- You are NOT required to call a tool on every turn. Most conversational queries do NOT need tools.\n"
    "- NEVER call any tool for questions about past conversations, user preferences, personal details, memory recall, greetings, or general chat. "
    "Answer those directly in plain conversational English without calling any tools.\n"
    "- ONLY call a tool if the user's CURRENT query directly and explicitly requests that specific tool's capability.\n"
    "- When past conversation reference is provided, use that information directly to answer the user. Do NOT call tools to look up or verify past conversation information.\n"
    "- FRESH TOOL RESULTS TAKE ABSOLUTE PRIORITY: When a tool is called in the current turn (such as get_current_datetime, web_search, run_python, set_reminder, etc.) and returns a result, you MUST answer the user using that fresh, live tool result. Fresh tool results ALWAYS override any past memory reference or previous conversation snippet. NEVER echo, substitute, or mix in past memory when a tool has just provided the fresh answer for the current query.\n"
    "- NEVER NARRATE TOOL CALLS OR OUTPUT RAW JSON: NEVER output narration phrases like \"I'll call the `web_search` tool\" or \"I will run python\" and NEVER output raw JSON tool-calling blocks like {\"name\": ...} in your conversational text. To call a tool, invoke it through the tool calling interface directly. In your final text response to the user, speak naturally in plain conversational English without mentioning tool names, parameters, or code syntax.\n\n"
    "CRITICAL TOOL INSTRUCTIONS:\n"
    "- `open_application`: Call this tool whenever the user asks to open, launch, or start an app. Always invoke the tool directly.\n"
    "- `get_current_datetime`: Call this tool for any questions regarding the current date, time, day of the week, month, or year. Always use get_current_datetime (never web_search) for date or time queries. Always synthesize into a single natural spoken sentence.\n"
    "- When and ONLY when `type_text` or `press_enter_key` was ACTUALLY executed and returned a tool result in the current turn: In your final response, you MUST state the EXACT window name reported in that tool result. "
    "If the tool result states that typing succeeded, you MUST confirm that typing succeeded into that window. "
    "If the tool result states that typing was cancelled, only then report that it was cancelled into that window. "
    "If the user asks to type text and send it or press enter, call `type_text` with `press_enter=True`. "
    "NEVER assume, guess, or invent a different window name. "
    "If NO tool was executed in this turn, you MUST NEVER mention typing, keystrokes, cancellations, or target windows.\n"
    "- When a tool returns an error (starts with 'Error:'), you MUST identify the SPECIFIC error type and cause reported by the tool, "
    "and quote or closely paraphrase the exact reason. NEVER use a generic 'stopped for exceeding the time limit' explanation unless the error is ACTUALLY an execution timeout.\n"
    "  * Security / Restricted imports (e.g. 'Error: SecurityError: Importing ... is prohibited'): Explain that the code was blocked because it tried to import a restricted module or execute prohibited operations.\n"
    "  * Execution timeouts (e.g. 'Error: Execution timed out'): State that the code was stopped because it took too long and exceeded the time limit (never describe a timeout as intended or successful).\n"
    "  * Syntax / Runtime exceptions (e.g. 'SyntaxError', 'ZeroDivisionError', 'PermissionError'): State the exact error (e.g. 'syntax error', 'division by zero', or 'filesystem access disabled').\n"
    "- NEVER INVENT OR FABRICATE TOOL RESULTS OR ACTIONS: NEVER invent, fabricate, or hallucinate a plausible result when a tool fails, returns an error, or was NOT called. If no tool was called in the current turn, you must NEVER claim or imply that any tool was executed, attempted, cancelled, rejected, or failed. Always base action descriptions strictly on actual tool results present in the current turn's message history.\n\n"
    "Keep your final responses concise, natural, and conversational — suitable for being spoken aloud, "
    "ideally under ~40 words unless the user explicitly asks for detail, an explanation, or a list. "
    "Avoid markdown formatting, headers, or bullet points unless specifically requested."
)


class AgentState(TypedDict):
    """
    Core state for the Jarvis agent.
    messages: List of conversation messages updated using the `add_messages` reducer.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_ollama_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    **kwargs: Any,
) -> ChatOllama:
    """
    Initializes and returns a ChatOllama LLM client connected to local Ollama.
    keep_alive="30m" ensures the model remains resident in GPU VRAM between queries.
    """
    llm_kwargs = {**kwargs}
    if num_predict is not None:
        llm_kwargs["num_predict"] = num_predict

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        keep_alive=keep_alive,
        **llm_kwargs,
    )


def extract_fallback_tool_calls(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Detects if the LLM outputted a tool call as raw JSON in plain text content
    instead of structured tool_calls (common with smaller 3B models or when narrating).

    Extracts the tool call dictionary and returns:
    (cleaned_content, tool_calls_list)
    """
    if not content or not isinstance(content, str):
        return content, []

    tool_calls: List[Dict[str, Any]] = []
    cleaned = content

    pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:parameters|arguments|args)"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    for m in matches:
        full_match = m.group(0)
        tool_name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            parsed_args = json.loads(raw_args)
        except Exception:
            try:
                parsed_args = json.loads(raw_args.replace("'", '"'))
            except Exception:
                parsed_args = {}

        # If web_search was called for math or percentage, redirect to run_python
        if tool_name == "web_search":
            q = str(parsed_args.get("query", "")).strip().lower()
            pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", q)
            if pm:
                pct = float(pm.group(1)) / 100.0
                val = float(pm.group(2))
                tool_name = "run_python"
                parsed_args = {"code": f"{val} * {pct}"}
            elif re.search(r"^\d+\s*[\+\-\*\/]\s*\d+", q):
                tool_name = "run_python"
                parsed_args = {"code": q}

        tool_calls.append({
            "name": tool_name,
            "args": parsed_args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        })
        cleaned = cleaned.replace(full_match, "").strip()

    # Clean markdown fences around removed json
    cleaned = re.sub(r"```(?:json)?\s*```", "", cleaned).strip()

    # Clean narration boilerplate like "To calculate ..., I'll call the `web_search` tool."
    cleaned = re.sub(
        r"(?:To (?:calculate|search|find|check|run) [^,.]*,\s*)?I(?:'ll| will)\s+(?:call|use|run|execute)\s+the\s+[`'\"]?\w+[`'\"]?\s+tool\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned, tool_calls


def extract_action_intent_fallback(user_query: str, model_content: str = "") -> List[Dict[str, Any]]:
    """
    If the LLM narrated an action (e.g. 'Opening Notepad.') or generated plain text
    instead of calling the tool for an unambiguous action command, synthesizes the
    appropriate tool call so the action is reliably executed rather than merely narrated.
    """
    if not user_query or not user_query.strip():
        return []

    clean_q = re.sub(r"[^\w\s\-\.%]", "", user_query).lower().strip()

    # 1. Application Launch: "open notepad", "launch chrome", "start calculator"
    app_match = re.match(r"^(?:please\s+)?(?:open|launch|start|run)\s+(?:the\s+)?([a-zA-Z0-9_\-\. ]+)$", clean_q)
    if app_match:
        app_name = app_match.group(1).strip()
        non_apps = {"a window", "the window", "the door", "a door", "my eyes", "this", "that", "it", "a file", "the file"}
        if app_name and app_name not in non_apps and len(app_name.split()) <= 4:
            return [{
                "name": "open_application",
                "args": {"app_name": app_name},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }]

    # 2. Desktop Typing: "type <text>" or "type <text> and press enter"
    type_match = re.match(r"^(?:please\s+)?type\s+(.+)$", clean_q)
    if type_match:
        raw_target = type_match.group(1).strip()
        press_enter = False
        if "and press enter" in raw_target:
            press_enter = True
            raw_target = raw_target.replace("and press enter", "").strip()
        elif "and send it" in raw_target or "and send" in raw_target:
            press_enter = True
            raw_target = re.sub(r"\s+and\s+send(?:\s+it)?", "", raw_target).strip()

        if raw_target:
            return [{
                "name": "type_text",
                "args": {"text": raw_target, "press_enter": press_enter},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }]

    # 3. Press Enter Key: "press enter", "hit enter", "send it"
    if clean_q in ("press enter", "hit enter", "press enter key", "send it", "send message"):
        return [{
            "name": "press_enter_key",
            "args": {},
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }]

    # 4. Datetime query: "what is the date", "what time is it", "what day is today"
    datetime_phrases = (
        "what time is it", "what's the time", "tell me the time", "what is the date",
        "what's the date", "what is today's date", "what date is today", "what day is today",
        "what day is it", "what is the current time", "what is the current date",
    )
    if any(clean_q == p or clean_q.startswith(f"{p} ") for p in datetime_phrases):
        return [{
            "name": "get_current_datetime",
            "args": {},
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }]

    # 5. Math / Percentage evaluation: "calculate 358% of 340", "what is 25 * 4"
    pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", clean_q)
    if pm:
        pct = float(pm.group(1)) / 100.0
        val = float(pm.group(2))
        return [{
            "name": "run_python",
            "args": {"code": f"{val} * {pct}"},
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }]

    return []


def is_tool_or_action_query(query: str) -> bool:
    """
    Determines whether a user query requires active tool execution (e.g. clock/datetime,
    math calculations, app launching, desktop typing, reminders, web search, or clipboard).
    Queries requiring real-time tool execution should never retrieve or be constrained by
    stale vector memory snippets from previous unrelated turns.
    """
    if not query or not query.strip():
        return False
    clean = query.strip().lower()

    # Datetime indicators
    datetime_indicators = (
        "time", "date", "clock", "day today", "today's day", "day of the week",
        "what day", "what date", "today", "current time", "current date",
        "tell me the time", "tell me the date", "what time", "what's the time",
    )
    if any(ind in clean for ind in datetime_indicators):
        return True

    # Mathematical calculations & percentages
    math_indicators = (
        "calculate", "compute", "math", "%", "percent", "percentage",
        "plus", "minus", "divided by", "multiplied by", "times",
    )
    if any(ind in clean for ind in math_indicators):
        return True
    if re.search(r"\d+\s*[\+\-\*\/\%]\s*\d+", clean) or re.search(r"\d+\s*(?:percent|%)\s*of\s*\d+", clean):
        return True

    # Desktop actions & typing (broad matching: startswith OR key action words)
    desktop_keywords = (
        "type", "typing", "typed", "write", "open", "launch", "start",
        "press enter", "hit enter", "send it", "send message", "close",
        "notepad", "chrome", "powershell", "whatsapp", "browser", "application",
    )
    if any(re.search(rf"\b{re.escape(kw)}\b", clean) for kw in desktop_keywords):
        return True

    # Clipboard
    if any(ind in clean for ind in ("clipboard", "copied")):
        return True

    # Reminders
    if any(ind in clean for ind in ("remind", "reminder", "alarm", "schedule")):
        return True

    # Web search for live external info
    search_indicators = ("search", "google", "look up", "find online", "weather")
    if any(ind in clean for ind in search_indicators):
        return True

    return False


def create_llm_node(
    llm: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enable_memory: bool = True,
    memory_db_path: Optional[str] = None,
):
    """
    Factory creating the llm_node for the LangGraph workflow.
    Ensures the system persona prompt is anchored at the start of context,
    enriches the prompt with relevant past context from ChromaDB vector memory,
    logs whether the Ollama call was 'cold' or 'warm', logs tool invocations,
    and stores completed conversational turns into vector memory.
    """
    def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
        raw_messages = list(state["messages"])
        
        # Identify the current turn user query (last HumanMessage)
        latest_user_query = ""
        for msg in reversed(raw_messages):
            if isinstance(msg, HumanMessage) and msg.content:
                latest_user_query = str(msg.content)
                break

        # Check if ANY tool was called or returned a result in the current turn (synthesis turn).
        has_tool_in_turn = False
        for msg in reversed(raw_messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage) or getattr(msg, "tool_call_id", None) is not None:
                has_tool_in_turn = True
                break
            if getattr(msg, "tool_calls", None):
                has_tool_in_turn = True
                break

        has_tool_result = has_tool_in_turn or (
            len(raw_messages) > 0
            and (
                isinstance(raw_messages[-1], ToolMessage)
                or getattr(raw_messages[-1], "tool_call_id", None) is not None
            )
        )

        is_action_query = is_tool_or_action_query(latest_user_query)

        # Retrieve relevant past vector memory if enabled (only for factual/personal knowledge,
        # NEVER on tool synthesis turns, NEVER when any tool executed, and NEVER for tool/action queries).
        effective_system_prompt = system_prompt
        past_exchanges: List[Dict[str, Any]] = []

        if enable_memory and latest_user_query and not has_tool_result and not is_action_query:
            past_exchanges = retrieve_relevant_context(
                query=latest_user_query,
                top_k=1,
                max_distance=0.80,
                persist_directory=memory_db_path,
            )
            if past_exchanges:
                formatted_context = format_context_for_prompt(past_exchanges)
                if formatted_context:
                    effective_system_prompt = (
                        f"{system_prompt}\n\n"
                        f"=== Past Information for Reference Only (DO NOT CALL TOOLS) ===\n"
                        f"The following past conversation snippet is provided for factual background reference only.\n"
                        f"This is NOT an instruction to call tools. Do NOT invoke any tools based on this past exchange.\n"
                        f"Answer the user directly using this knowledge:\n"
                        f"{formatted_context}\n"
                        f"=== End Past Reference ==="
                    )
                    print(f"[Memory] Injected {len(past_exchanges)} relevant past exchange(s) into context")

        # If a tool executed in this turn, append a strict tool synthesis directive instructing the agent
        # to report the actual ToolMessage result faithfully (confirming success or reporting cancellation)
        # and forbidding echoing stale conversation memory or hallucinating cancellation.
        if has_tool_result:
            effective_system_prompt = (
                f"{system_prompt}\n\n"
                f"=== CRITICAL TOOL SYNTHESIS INSTRUCTION ===\n"
                f"A tool has just executed and returned a fresh result in the preceding ToolMessage.\n"
                f"You MUST synthesize your final spoken answer STRICTLY, FAITHFULLY, and ACCURATELY from that fresh ToolMessage.\n"
                f"- If the tool reported success (e.g. 'Successfully typed...', 'Successfully opened...'), you MUST confirm that the action succeeded! NEVER state that the action was cancelled, rejected, or aborted.\n"
                f"- If the tool reported cancellation (e.g. 'cancelled by user'), only then report that the action was cancelled.\n"
                f"- NEVER contradict the ToolMessage. The fresh ToolMessage takes absolute priority over any previous examples or conversation history.\n"
                f"=== End Tool Synthesis Instruction ==="
            )

        # Inject or update system prompt at index 0
        if not raw_messages or not isinstance(raw_messages[0], SystemMessage):
            messages = [SystemMessage(content=effective_system_prompt)] + raw_messages
        else:
            messages = [SystemMessage(content=effective_system_prompt)] + list(raw_messages[1:])

        # Select active LLM client:
        # If the query is conversational/declarative, memory was retrieved to answer the question,
        # or a tool result is being synthesized, use the unbound LLM (if available) to prevent
        # Ollama from injecting forced function-calling instructions on non-action queries or synthesis turns.
        # NEVER unbind tools for action queries (e.g. datetime, math, reminders, desktop, search).
        active_llm = llm
        unbound_llm = llm.bound if isinstance(llm, RunnableBinding) else None
        if unbound_llm is not None:
            if has_tool_result:
                # Tools have already executed in this turn; synthesize the result directly without tool-calling templates
                active_llm = unbound_llm
            elif latest_user_query and not is_action_query:
                clean_q = latest_user_query.strip().lower()
                declarative_prefixes = (
                    "my favorite", "my name", "my job", "my work", "my hobby", "my dog", "my cat",
                    "i like", "i love", "i live", "i work", "i am", "i'm", "i prefer", "i enjoy",
                    "i have", "i want to tell you", "just so you know", "remember that",
                    "hello", "hi", "hey", "good morning", "good evening", "how are you",
                    "thank you", "thanks",
                )
                is_declarative = any(clean_q.startswith(p) for p in declarative_prefixes)
                has_memory = bool(past_exchanges)
                if has_memory or is_declarative:
                    active_llm = unbound_llm

        response = active_llm.invoke(messages)

        # Inspect Ollama response metadata to log cold vs warm status
        raw_meta = getattr(response, "response_metadata", None)
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        model_name = metadata.get("model") or getattr(llm, "model", "unknown")
        load_duration_ns = metadata.get("load_duration", 0)
        if isinstance(load_duration_ns, (int, float)) and load_duration_ns > 0:
            load_duration_s = load_duration_ns / 1e9
            # Over 0.5s indicates model weights had to be loaded from storage to VRAM
            if load_duration_s >= 0.5:
                call_type = f"COLD call (model load required: {load_duration_s:.2f}s)"
            else:
                call_type = f"WARM call (model resident in VRAM: {load_duration_s * 1000:.1f}ms)"
            print(f"[Ollama ({model_name})] {call_type}")

        # Check if the model requested any tool calls and log them
        has_tool_calls = bool(hasattr(response, "tool_calls") and response.tool_calls)
        if has_tool_calls:
            for tc in response.tool_calls:
                # Math re-routing guard for native tool calls
                if tc.get("name") == "web_search":
                    raw_q = str(tc.get("args", {}).get("query", "")).strip().lower()
                    pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", raw_q)
                    if pm:
                        pct = float(pm.group(1)) / 100.0
                        val = float(pm.group(2))
                        tc["name"] = "run_python"
                        tc["args"] = {"code": f"{val} * {pct}"}
                    elif re.search(r"^\d+\s*[\+\-\*\/]\s*\d+", raw_q):
                        tc["name"] = "run_python"
                        tc["args"] = {"code": raw_q}

                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                print(f"[Tool Invoked by Agent] {tool_name}(args={tool_args})")
        else:
            # Fallback: Detect if the model outputted a tool call as raw JSON text
            if getattr(response, "content", None):
                raw_content = str(response.content)
                cleaned_content, fallback_tool_calls = extract_fallback_tool_calls(raw_content)
                if fallback_tool_calls:
                    response.tool_calls = fallback_tool_calls
                    response.content = cleaned_content
                    has_tool_calls = True
                    for tc in response.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        print(f"[Tool Invoked via Fallback Parser] {tool_name}(args={tool_args})")

        if not has_tool_calls:
            # Fallback 2: Check if user query was an explicit action command that the model narrated instead of invoking
            if latest_user_query and not has_tool_result:
                intent_tool_calls = extract_action_intent_fallback(latest_user_query, str(getattr(response, "content", "")))
                if intent_tool_calls:
                    response.tool_calls = intent_tool_calls
                    response.content = ""
                    has_tool_calls = True
                    for tc in response.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        print(f"[Tool Invoked via Intent Fallback] {tool_name}(args={tool_args})")

        if not has_tool_calls:
            # Final assistant response produced (no tool calls pending)
            # Store turn in vector memory only if NO tool was executed in this turn and query is not an action query
            if enable_memory and latest_user_query and not has_tool_result and not is_action_query and getattr(response, "content", None):
                content_str = str(response.content).strip()
                if content_str and not content_str.startswith("Error:"):
                    store_exchange(
                        user_message=latest_user_query,
                        assistant_response=content_str,
                        persist_directory=memory_db_path,
                    )
                    print("[Memory] Stored conversation exchange into vector memory")

        return {"messages": [response]}

    return llm_node


def build_graph(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Optional[Sequence[Any]] = None,
    enable_memory: bool = True,
    memory_db_path: Optional[str] = None,
):
    """
    Constructs and compiles the LangGraph agent state machine:
      START -> llm_node -> (tools_condition: has tool_calls? -> tools_node -> llm_node : END)

    Returns a compiled LangGraph Runnable configured for the specified model and tools.
    """
    if tools is None:
        tools = list(DEFAULT_TOOLS)

    llm = get_ollama_llm(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        keep_alive=keep_alive,
    )
    bound_llm = llm.bind_tools(tools) if tools else llm
    llm_node = create_llm_node(
        llm=bound_llm,
        system_prompt=system_prompt,
        enable_memory=enable_memory,
        memory_db_path=memory_db_path,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_edge(START, "llm")

    if tools:
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_conditional_edges("llm", tools_condition)
        workflow.add_edge("tools", "llm")
    else:
        workflow.add_edge("llm", END)

    return workflow.compile()


# Module-level compiled graph cache: cache_key -> CompiledStateGraph
_AGENT_APPS: Dict[str, Any] = {}


def get_agent_app(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Optional[Sequence[Any]] = None,
    enable_memory: bool = True,
    memory_db_path: Optional[str] = None,
):
    """
    Returns a cached compiled agent graph instance for the specified model configuration.
    Avoids recompiling on every turn while correctly isolating different configurations in memory.
    """
    tools_tuple = tuple(tools) if tools is not None else tuple(DEFAULT_TOOLS)
    tools_key = tuple(getattr(t, "name", str(t)) for t in tools_tuple)
    cache_key = (
        f"{model}::{base_url}::{temperature}::{num_predict}::{keep_alive}::"
        f"{hash(system_prompt)}::{tools_key}::{enable_memory}::{memory_db_path}"
    )
    if cache_key not in _AGENT_APPS:
        _AGENT_APPS[cache_key] = build_graph(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
            keep_alive=keep_alive,
            system_prompt=system_prompt,
            tools=tools_tuple,
            enable_memory=enable_memory,
            memory_db_path=memory_db_path,
        )
    return _AGENT_APPS[cache_key]


def get_default_agent_app(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    tools: Optional[Sequence[Any]] = None,
    enable_memory: bool = True,
    memory_db_path: Optional[str] = None,
):
    """Backwards-compatible alias for get_agent_app."""
    return get_agent_app(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        keep_alive=keep_alive,
        tools=tools,
        enable_memory=enable_memory,
        memory_db_path=memory_db_path,
    )


def run_agent(
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    app=None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tools: Optional[Sequence[Any]] = None,
    enable_memory: bool = True,
    memory_db_path: Optional[str] = None,
) -> Tuple[str, List[BaseMessage]]:
    """
    Executes a single conversational turn through the LangGraph agent.

    :param user_input: Text prompt from the user.
    :param history: List of preceding BaseMessage instances (None starts a new conversation).
    :param app: Pre-compiled LangGraph application (if None, uses/creates graph for the specified model).
    :param model: Ollama model name (e.g. 'llama3.1:8b', 'llama3.2:3b').
    :param base_url: Ollama API endpoint.
    :param temperature: LLM sampling temperature.
    :param num_predict: Optional token limit cap for response generation.
    :param keep_alive: Time duration to keep model loaded in VRAM (default: "30m").
    :param system_prompt: System prompt for the agent persona.
    :param tools: Sequence of tools to bind to the agent (defaults to DEFAULT_TOOLS).
    :param enable_memory: Whether to enable persistent vector memory retrieval and storage.
    :param memory_db_path: Optional directory path for ChromaDB vector store.
    :return: (assistant_response_text, updated_message_history)
    """
    if app is None:
        app = get_agent_app(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
            keep_alive=keep_alive,
            system_prompt=system_prompt,
            tools=tools,
            enable_memory=enable_memory,
            memory_db_path=memory_db_path,
        )

    current_messages: List[BaseMessage] = list(history) if history else []
    current_messages.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": current_messages})
    all_messages: List[BaseMessage] = list(result["messages"])

    # Extract latest AI message response
    last_message = all_messages[-1]
    raw_response_text = str(last_message.content) if last_message else ""
    response_text = sanitize_speech_text(raw_response_text)

    return response_text, all_messages

