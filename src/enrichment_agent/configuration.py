"""Configuration schema for the HERMES agent runtime."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:

    # --- Selección de Modelo ---
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        metadata={
            "description": "LLM identifier used across all SEC nodes and convergence."
        },
    )

    # --- Límites de Herramientas ---
    max_search_results: int = field(
        default=3,
        metadata={
            "description": "Maximum number of search results per query (SEC 1 only)."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Load runtime configuration from a RunnableConfig."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})