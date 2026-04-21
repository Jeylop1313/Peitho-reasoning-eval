"""
Tools for the Relevance node (SEC 1).
Optimized for Groq free tier (30K TPM).

Two tools: search (general) and search_news (recent news).
Both use search_depth="basic" for short NLP summaries.
search_news is inactive by default — set USE_NEWS_SEARCH=True
in Graph.py to enable it for real-time datasets.
"""

from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langchain_tavily import TavilySearch
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration


@tool
async def search(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """General web search.

    Useful for fact-checking, finding information about events, people, or topics.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearch(
        max_results=configuration.max_search_results,
        search_depth="basic",
    )
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


@tool
async def search_news(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Recent news search.

    Useful for finding current events or recent developments.
    Activate via USE_NEWS_SEARCH=True in Graph.py.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearch(
        max_results=configuration.max_search_results,
        search_depth="basic",
        topic="news",
        time_range="week",
    )
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


__all__ = ["search", "search_news"]