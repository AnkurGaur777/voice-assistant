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

# --- Default Configuration ---
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.7

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, capable, and intelligent local voice assistant. "
    "Keep your responses concise, natural, and conversational — suitable for being spoken aloud, "
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
    **kwargs: Any,
) -> ChatOllama:
    """
    Initializes and returns a ChatOllama LLM client connected to local Ollama.
    """
    llm_kwargs = {**kwargs}
    if num_predict is not None:
        llm_kwargs["num_predict"] = num_predict

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **llm_kwargs,
    )


def create_llm_node(llm: ChatOllama, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """
    Factory creating the llm_node for the LangGraph workflow.
    Ensures the system persona prompt is anchored at the start of context.
    """
    def llm_node(state: AgentState) -> Dict[str, List[BaseMessage]]:
        raw_messages = list(state["messages"])
        
        # Inject system prompt if not present
        if not raw_messages or not isinstance(raw_messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + raw_messages
        else:
            messages = raw_messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_graph(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """
    Constructs and compiles the minimal LangGraph agent state machine:
      START -> llm_node -> END

    Returns a compiled LangGraph Runnable.
    """
    llm = get_ollama_llm(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
    )
    llm_node = create_llm_node(llm=llm, system_prompt=system_prompt)

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()


# Module-level default compiled graph singleton
_DEFAULT_AGENT_APP = None


def get_default_agent_app(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: Optional[int] = None,
):
    """
    Returns a cached compiled agent graph instance to avoid recompiling on every turn.
    """
    global _DEFAULT_AGENT_APP
    if _DEFAULT_AGENT_APP is None:
        _DEFAULT_AGENT_APP = build_graph(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
        )
    return _DEFAULT_AGENT_APP


def run_agent(
    user_input: str,
    history: Optional[List[BaseMessage]] = None,
    app=None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    num_predict: Optional[int] = None,
) -> Tuple[str, List[BaseMessage]]:
    """
    Executes a single conversational turn through the LangGraph agent.

    :param user_input: Text prompt from the user.
    :param history: List of preceding BaseMessage instances (None starts a new conversation).
    :param app: Pre-compiled LangGraph application (uses default if None).
    :param model: Ollama model name.
    :param base_url: Ollama API endpoint.
    :param num_predict: Optional token limit cap for response generation.
    :return: (assistant_response_text, updated_message_history)
    """
    if app is None:
        app = get_default_agent_app(model=model, base_url=base_url, num_predict=num_predict)

    current_messages: List[BaseMessage] = list(history) if history else []
    current_messages.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": current_messages})
    all_messages: List[BaseMessage] = list(result["messages"])

    # Extract latest AI message response
    last_message = all_messages[-1]
    response_text = str(last_message.content) if last_message else ""

    return response_text, all_messages
