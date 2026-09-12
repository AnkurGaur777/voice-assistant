"""
Local Jarvis - LangGraph Brain & LLM Orchestration Module (Phase 3)

This module builds the core LangGraph state machine connecting to a local Ollama
instance (model: llama3.1:8b, base_url: http://localhost:11434) running on GPU.

Architecture:
  START -> llm_node -> END
  (Designed as a modular, extensible foundation ready for tool_node integration in Phase 4)
"""

import os
import sys
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.tools.clipboard import read_clipboard, summarize_clipboard
from src.agent.tools.datetime_tool import get_current_datetime
from src.agent.tools.desktop import open_application, type_text
from src.agent.tools.web_search import web_search

# --- Default Configuration ---
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TOOLS = [
    get_current_datetime,
    web_search,
    read_clipboard,
    summarize_clipboard,
    open_application,
    type_text,
]

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, capable, and intelligent local voice assistant.\n"
    "You have access to the following tools:\n"
    "1. `get_current_datetime`: Call this tool for any questions regarding the current date, time, "
    "day of the week, month, or year. Always use get_current_datetime (never web_search) for date or time queries.\n"
    "2. `web_search`: Call this tool to look up live external information, such as breaking news, weather, "
    "sports scores, or events beyond your training data.\n"
    "3. `read_clipboard`: Call this tool when the user asks to read, inspect, or check what is currently copied "
    "to the system clipboard, or asks specific questions about copied text.\n"
    "4. `summarize_clipboard`: Call this tool when the user asks to summarize, give an overview, or highlight "
    "key points of text currently copied to the system clipboard.\n"
    "5. `open_application`: Call this tool when the user asks to open, launch, or start a program or application "
    "on their computer (e.g. notepad, calculator, browser, terminal). Always check the tool result to report whether the window was brought to focus.\n"
    "6. `type_text`: Call this tool when the user asks to type or enter text into the active desktop window. "
    "Extract all words or sentences following verbs like 'type', 'write', or 'enter' as the `text` argument "
    "(for example: 'type this should not appear' -> text='this should not appear'). "
    "CRITICAL: In your final response, you MUST state the EXACT window name reported in the tool result "
    "(e.g. if the tool result mentions 'Windows PowerShell', you must report 'Windows PowerShell'). "
    "NEVER assume, guess, or invent the window name (e.g. do NOT say 'Notepad' if the tool reported 'Windows PowerShell').\n"
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


def create_llm_node(llm: Any, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """
    Factory creating the llm_node for the LangGraph workflow.
    Ensures the system persona prompt is anchored at the start of context,
    logs whether the Ollama call was 'cold' or 'warm', and logs any tool invocations.
    """
    def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
        raw_messages = list(state["messages"])
        
        # Inject system prompt if not present
        if not raw_messages or not isinstance(raw_messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + raw_messages
        else:
            messages = raw_messages

        response = llm.invoke(messages)

        # Inspect Ollama response metadata to log cold vs warm status
        metadata = getattr(response, "response_metadata", {}) or {}
        model_name = metadata.get("model") or getattr(llm, "model", "unknown")
        load_duration_ns = metadata.get("load_duration", 0)
        if load_duration_ns is not None:
            load_duration_s = load_duration_ns / 1e9
            # Over 0.5s indicates model weights had to be loaded from storage to VRAM
            if load_duration_s >= 0.5:
                call_type = f"COLD call (model load required: {load_duration_s:.2f}s)"
            else:
                call_type = f"WARM call (model resident in VRAM: {load_duration_s * 1000:.1f}ms)"
            print(f"[Ollama ({model_name})] {call_type}")

        # Check if the model requested any tool calls and log them
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                print(f"[Tool Invoked by Agent] {tool_name}(args={tool_args})")

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
    llm_node = create_llm_node(llm=bound_llm, system_prompt=system_prompt)

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
):
    """
    Returns a cached compiled agent graph instance for the specified model configuration.
    Avoids recompiling on every turn while correctly isolating different configurations in memory.
    """
    tools_tuple = tuple(tools) if tools is not None else tuple(DEFAULT_TOOLS)
    tools_key = tuple(getattr(t, "name", str(t)) for t in tools_tuple)
    cache_key = (
        f"{model}::{base_url}::{temperature}::{num_predict}::{keep_alive}::"
        f"{hash(system_prompt)}::{tools_key}"
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
        )
    return _AGENT_APPS[cache_key]


def get_default_agent_app(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    tools: Optional[Sequence[Any]] = None,
):
    """Backwards-compatible alias for get_agent_app."""
    return get_agent_app(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        keep_alive=keep_alive,
        tools=tools,
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
        )

    current_messages: List[BaseMessage] = list(history) if history else []
    current_messages.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": current_messages})
    all_messages: List[BaseMessage] = list(result["messages"])

    # Extract latest AI message response
    last_message = all_messages[-1]
    response_text = str(last_message.content) if last_message else ""

    return response_text, all_messages
