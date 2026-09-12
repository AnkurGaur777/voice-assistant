"""
Local Jarvis - Web Search Tool (Phase 4a)

Queries the local SearXNG instance (http://localhost:8080/search?q=...&format=json)
and formats the top results (title, url, snippet) as a clean string for the LLM.
"""

import os
from typing import Optional
import requests
from langchain_core.tools import tool

DEFAULT_SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")
DEFAULT_MAX_RESULTS = 4
DEFAULT_TIMEOUT_SECONDS = 6.0


def search_searxng(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    base_url: str = DEFAULT_SEARXNG_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Executes a search query against the local SearXNG instance.

    :param query: Search query string.
    :param max_results: Number of top results to return (default 4, range 3-5).
    :param base_url: SearXNG base URL (default: http://localhost:8080).
    :param timeout: HTTP request timeout in seconds.
    :return: Formatted text string containing top search results or error message.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Search query cannot be empty."

    search_endpoint = f"{base_url.rstrip('/')}/search"
    params = {
        "q": clean_query,
        "format": "json",
    }

    print(f"[Search Engine] Querying SearXNG for: '{clean_query}'...")
    try:
        response = requests.get(search_endpoint, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        err_msg = (
            f"Error: Unable to connect to local SearXNG service at {base_url}. "
            "Ensure the SearXNG Docker container ('local-jarvis-searxng') is running."
        )
        print(f"[Search Engine] {err_msg}")
        return err_msg
    except requests.exceptions.Timeout:
        err_msg = f"Error: Search request to {base_url} timed out after {timeout} seconds."
        print(f"[Search Engine] {err_msg}")
        return err_msg
    except requests.exceptions.RequestException as req_err:
        err_msg = f"Error: Search query failed ({req_err})."
        print(f"[Search Engine] {err_msg}")
        return err_msg
    except ValueError:
        err_msg = "Error: Failed to parse JSON response from SearXNG search service."
        print(f"[Search Engine] {err_msg}")
        return err_msg

    results = data.get("results", [])
    if not results:
        suggestions = data.get("suggestions", [])
        suggestion_text = f" Suggestions: {', '.join(suggestions)}" if suggestions else ""
        print(f"[Search Engine] No results found for: '{clean_query}'.")
        return f"No search results found for: '{clean_query}'.{suggestion_text}"

    top_results = results[:max_results]
    print(f"[Search Engine] Retrieved {len(results)} results (returning top {len(top_results)} to LLM).")
    formatted_items = []

    for idx, item in enumerate(top_results, start=1):
        title = item.get("title", "Untitled").strip()
        url = item.get("url", "").strip()
        content = item.get("content", "").strip() or item.get("snippet", "").strip()

        snippet = " ".join(content.split()) if content else "No snippet available."

        formatted_items.append(
            f"[{idx}] {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}"
        )

    header = f'Web search results for "{clean_query}":'
    return f"{header}\n\n" + "\n\n".join(formatted_items)


@tool
def web_search(query: str) -> str:
    """Search the web for live, external, or real-time information.

    Use this tool ONLY when the user asks to search the web or look up live external
    information on the internet (such as latest news, weather, sports scores, public web facts).
    NEVER call this tool for:
    - Math, arithmetic, percentages, or numerical calculations (use run_python instead).
    - Questions about the user, user preferences, past conversations, or private details.

    Args:
        query: The search term or keywords to query the search engine with.
    """
    clean_q = (query or "").strip().lower()
    # Guard: If query is purely a percentage or arithmetic calculation, delegate to sandbox
    import re
    if re.search(r"\d+\s*(?:%|percent)\s*of\s*\d+", clean_q) or re.search(r"^\d+\s*[\+\-\*\/]\s*\d+", clean_q):
        from src.agent.tools.sandbox import run_python
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", clean_q)
        if m:
            pct = float(m.group(1)) / 100.0
            val = float(m.group(2))
            return run_python.invoke({"code": f"{val} * {pct}"})
        else:
            return run_python.invoke({"code": clean_q})

    return search_searxng(query=query, max_results=DEFAULT_MAX_RESULTS)
