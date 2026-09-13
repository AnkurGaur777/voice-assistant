"""
Local Jarvis - Local Vector Memory with ChromaDB (Phase 6)

Provides persistent local vector memory using ChromaDB.
Exchanges are embedded with ChromaDB's lightweight CPU-based ONNX model
(all-MiniLM-L6-v2, 384 dimensions) and stored on disk in chroma_db/.
Does not call Ollama or consume GPU VRAM, keeping inference fast and independent.
"""

import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb

# Ensure project root is determined
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "conversation_memory"

# In-memory client cache: resolved_path -> PersistentClient
_CLIENT_CACHE: Dict[str, chromadb.PersistentClient] = {}


def get_memory_client(
    persist_directory: Optional[Union[str, Path]] = None,
) -> chromadb.PersistentClient:
    """
    Returns a cached or new chromadb.PersistentClient targeting the specified directory.
    Defaults to PROJECT_ROOT / 'chroma_db'.
    """
    target_path = Path(persist_directory).resolve() if persist_directory else DEFAULT_DB_DIR.resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    cache_key = str(target_path)

    if cache_key not in _CLIENT_CACHE:
        # Initializing persistent client on disk
        client = chromadb.PersistentClient(path=cache_key)
        _CLIENT_CACHE[cache_key] = client

    return _CLIENT_CACHE[cache_key]


def get_memory_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[Union[str, Path]] = None,
) -> chromadb.Collection:
    """
    Returns or creates the ChromaDB collection using cosine similarity and
    the default lightweight CPU ONNX embedding function (all-MiniLM-L6-v2).
    """
    if client is None:
        client = get_memory_client(persist_directory=persist_directory)

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def store_exchange(
    user_message: str,
    assistant_response: str,
    metadata: Optional[Dict[str, Any]] = None,
    client: Optional[chromadb.PersistentClient] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    doc_id: Optional[str] = None,
) -> Optional[str]:
    """
    Embeds and persists a conversation turn (user message + assistant response)
    along with an ISO-8601 UTC timestamp and metadata.

    :param user_message: The text input from the user.
    :param assistant_response: The response produced by the assistant.
    :param metadata: Optional extra metadata dictionary.
    :param client: Optional pre-existing PersistentClient.
    :param persist_directory: Path to storage directory.
    :param collection_name: Name of ChromaDB collection.
    :param doc_id: Optional explicit document ID.
    :return: The stored document ID, or None if inputs were blank.
    """
    user_clean = (user_message or "").strip()
    assistant_clean = (assistant_response or "").strip()

    if not user_clean or not assistant_clean:
        return None

    # Do not store raw errors or tool failure boilerplate into vector memory
    if assistant_clean.startswith("Error:"):
        return None

    lower_resp = assistant_clean.lower()
    lower_user = user_clean.lower()

    # Do not persist ephemeral real-time datetime queries into permanent vector memory
    # Storing answers to "what time is it" poisons memory with stale timestamps
    ephemeral_time_queries = [
        "what time",
        "what's the time",
        "what is the time",
        "current time",
        "what time is it right now",
        "what day is it today",
        "what is today's date",
        "what's today's date",
        "what is the date",
        "what's the date",
        "date of today",
        "tell me the date",
        "tell me the time",
        "current date",
        "date today",
        "what day is today",
    ]
    if any(q in lower_user for q in ephemeral_time_queries):
        return None

    # Do not persist ephemeral desktop action / tool queries into permanent vector memory
    ephemeral_action_queries = [
        "type ", "typed", "typing",
        "open notepad", "open application", "launch ", "start ",
        "press enter", "hit enter", "send it", "send message",
        "write in notepad", "write into",
    ]
    if any(q in lower_user for q in ephemeral_action_queries):
        return None

    # Do not persist exchanges where the assistant expressed failure, lack of information, or uncertainty
    # to avoid negative feedback loops in future retrievals
    unpersisted_patterns = [
        "i don't have any information",
        "i do not have any information",
        "i don't have information",
        "i do not have information",
        "no information about",
        "i don't know",
        "i do not know",
        "i didn't recognize",
        "i did not recognize",
        "i couldn't",
        "i could not",
        "i'm not sure",
        "i am not sure",
        "cannot find",
        "could not find",
        "unable to find",
        "i apologize",
        "sorry, i",
    ]
    if any(pattern in lower_resp for pattern in unpersisted_patterns):
        return None

    # Do not persist corrupted, incoherent, or cross-tool contaminated responses
    corrupted_patterns = [
        "typing was cancelled",
        "typing was canceled",
        "typing was rejected",
        "typing was aborted",
        "typing cancelled",
        "typing canceled",
        "typing rejected",
        "typing aborted",
        "cancelled into",
        "canceled into",
        "aborted into",
        "rejected into",
        "successfully typed",
        "characters into",
        "character into",
        "keystrokes were sent",
        "keystrokes sent",
        "press enter key",
        "pressed enter",
        "pressing enter",
        "opening notepad",
        "opened notepad",
        "opening application",
        "opened application",
        "launching application",
        "launched application",
        "current system date and time",
        "current system date",
        "system date and time",
        "| - date:",
        "| - time:",
        "traceback (most recent call last)",
        "securityerror",
        "syntaxerror",
        "zerodivisionerror",
        "[tool invoked",
        "[clipboard]",
        "[sandbox]",
        "[system clock]",
        "[reminders]",
        '{"name":',
        '"parameters":',
        '"arguments":',
        "i'll call the",
        "i will call the",
        "calling the tool",
        "call the `",
    ]
    if any(pattern in lower_resp for pattern in corrupted_patterns):
        return None

    # Desktop action outcome phrases, character counts, and cancellations
    if re.search(r"typed\s+\d+\s+characters?", lower_resp):
        return None
    if re.search(r"typing\s+(?:was\s+)?(?:cancelled|canceled|rejected|aborted)", lower_resp):
        return None
    if re.search(r"successfully\s+typed", lower_resp):
        return None
    if re.search(r"(?:pressed|pressing)\s+enter", lower_resp):
        return None
    if re.search(r"keystrokes?\s+(?:were\s+)?(?:sent|injected|cancelled|canceled)", lower_resp):
        return None
    if re.search(r"(?:opened|opening|launched|launching)\s+[a-z0-9_\-\. ]+", lower_resp):
        return None

    # Mismatched cross-tool fragments (e.g. typing into date/time/clock/system)
    if re.search(r"typing.*into.*(date|time|clock|system)", lower_resp):
        return None

    collection = get_memory_collection(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    record_id = doc_id or f"exchange_{uuid.uuid4().hex}"
    document_text = f"User: {user_clean}\nAssistant: {assistant_clean}"

    meta: Dict[str, Any] = {
        "timestamp": now_iso,
        "user_message": user_clean,
        "assistant_response": assistant_clean,
    }
    if metadata:
        meta.update(metadata)

    collection.add(
        ids=[record_id],
        documents=[document_text],
        metadatas=[meta],
    )
    return record_id


def retrieve_relevant_context(
    query: str,
    top_k: int = 1,
    max_distance: float = 0.80,
    client: Optional[chromadb.PersistentClient] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Embeds the current query and returns the top-k most relevant past exchanges,
    filtering out items exceeding max_distance to prevent irrelevant context injection.

    :param query: Search query or user prompt.
    :param top_k: Maximum number of relevant exchanges to return (default: 1).
    :param max_distance: Cosine distance threshold (default: 0.80). Results with distance > max_distance are excluded.
    :param client: Optional pre-existing PersistentClient.
    :param persist_directory: Path to storage directory.
    :param collection_name: Name of ChromaDB collection.
    :return: List of dicts with keys: id, text, user_message, assistant_response, timestamp, distance.
    """
    query_clean = (query or "").strip()
    if not query_clean:
        return []

    collection = get_memory_collection(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    total_count = collection.count()
    if total_count == 0:
        print(f"[Memory Debug] Query: '{query_clean}' | Memory collection is empty (0 records).")
        return []

    n_results = min(max(1, top_k), total_count)
    results = collection.query(
        query_texts=[query_clean],
        n_results=n_results,
    )

    exchanges: List[Dict[str, Any]] = []
    if results and results.get("documents") and results["documents"]:
        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        if docs and distances:
            top_dist = distances[0]
            top_meta = metas[0] if metas else {}
            top_usr = top_meta.get("user_message", "")
            top_ast = top_meta.get("assistant_response", "")
            is_filtered = bool(top_dist is not None and top_dist > max_distance)
            status = f"FILTERED (exceeds {max_distance:.2f})" if is_filtered else f"ACCEPTED (<= {max_distance:.2f})"
            print(f"[Memory Debug] Top candidate distance: {top_dist:.4f} ({status}, threshold: {max_distance:.2f}) for query '{query_clean}' -> User: '{top_usr}' | Assistant: '{top_ast}'")

        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            dist = distances[i] if i < len(distances) else None
            record_id = ids[i] if i < len(ids) else None

            # Skip items that are semantically irrelevant
            if dist is not None and dist > max_distance:
                continue

            exchanges.append({
                "id": record_id,
                "text": doc,
                "user_message": meta.get("user_message", ""),
                "assistant_response": meta.get("assistant_response", ""),
                "timestamp": meta.get("timestamp", ""),
                "distance": dist,
            })

    return exchanges


def _sanitize_assistant_text(text: str) -> str:
    """
    Strips tool invocation logs, JSON tool calls, and error artifacts
    from assistant response text to keep memory purely conversational.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Exclude if the response is solely a raw error
    if cleaned.startswith("Error:"):
        return ""

    # Strip tool logging tags (e.g. [Tool Invoked...], [Clipboard], [Desktop Automation], etc.)
    cleaned = re.sub(r"\[(Tool Invoked|Clipboard|Desktop Automation|System Clock|Sandbox|Reminders|Memory|Ollama)[^\]]*\]\s*", "", cleaned)

    # Strip JSON function calls {"name": ..., "parameters": ...}
    cleaned = re.sub(r'\{"name":\s*"[^"]+",\s*"parameters":\s*\{[^}]*\}\}', '', cleaned)

    lower = cleaned.lower()
    # If the response indicates failure, uncertainty, cancellation boilerplate, or corrupted fragments, drop it
    unpersisted_snippets = [
        "i didn't recognize that as a valid time expression",
        "i don't see any text copied to the clipboard",
        "the clipboard is currently empty",
        "typing was cancelled",
        "typing was canceled",
        "typing was rejected",
        "typing was aborted",
        "typing cancelled",
        "typing canceled",
        "typing rejected",
        "typing aborted",
        "cancelled into",
        "canceled into",
        "aborted into",
        "rejected into",
        "successfully typed",
        "characters into",
        "character into",
        "keystrokes were sent",
        "keystrokes sent",
        "press enter key",
        "pressed enter",
        "pressing enter",
        "opening notepad",
        "opened notepad",
        "opening application",
        "opened application",
        "launching application",
        "launched application",
        "current system date and time",
        "current system date",
        "system date and time",
        "i don't have any information",
        "i do not have any information",
        "no information about",
        "i'm not sure",
        "i am not sure",
        "i couldn't",
        "i could not",
        "cannot find",
        "could not find",
        "unable to find",
        '{"name":',
        '"parameters":',
        '"arguments":',
        "i'll call the",
        "i will call the",
        "calling the tool",
        "call the `",
    ]
    if any(snippet in lower for snippet in unpersisted_snippets):
        return ""

    # Drop any desktop action outcome phrases, character count confirmations, or cancellations
    if re.search(r"typed\s+\d+\s+characters?", lower):
        return ""
    if re.search(r"typing\s+(?:was\s+)?(?:cancelled|canceled|rejected|aborted)", lower):
        return ""
    if re.search(r"successfully\s+typed", lower):
        return ""
    if re.search(r"(?:pressed|pressing)\s+enter", lower):
        return ""
    if re.search(r"keystrokes?\s+(?:were\s+)?(?:sent|injected|cancelled|canceled)", lower):
        return ""
    if re.search(r"(?:opened|opening|launched|launching)\s+[a-z0-9_\-\. ]+", lower):
        return ""

    if re.search(r"typing.*into.*(date|time|clock|system)", lower):
        return ""

    return cleaned.strip()


def format_context_for_prompt(context_items: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved memory exchanges into a readable block to inject
    into the agent's system prompt context.
    Excludes tool-call details, tool log prefixes, and tool failure artifacts,
    keeping only clean user messages and assistant conversational responses.
    """
    if not context_items:
        return ""

    formatted_entries: List[str] = []
    for item in context_items:
        u = item.get("user_message", "").strip()
        raw_a = item.get("assistant_response", "").strip()

        # If user_message is empty, try to extract from doc text
        if not u and item.get("text"):
            doc_text = item["text"]
            if "User: " in doc_text and "\nAssistant: " in doc_text:
                parts = doc_text.split("\nAssistant: ", 1)
                u = parts[0].replace("User: ", "").strip()
                if not raw_a:
                    raw_a = parts[1].strip()

        a = _sanitize_assistant_text(raw_a)
        if u and a:
            formatted_entries.append(f"- Past User: {u}\n  Past Assistant: {a}")

    return "\n".join(formatted_entries)


def clear_memory(
    client: Optional[chromadb.PersistentClient] = None,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """
    Deletes and recreates the memory collection, resetting stored vector memory.
    """
    if client is None:
        client = get_memory_client(persist_directory=persist_directory)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
