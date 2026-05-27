"""
HERMES — Data Contract (Scherer CPM)
======================================
Shared agent state and Convergence registry.

Architecture (sequential):
- SEC 1 Relevance    → free-text reasoning (str)
- SEC 2 Implication  → free-text reasoning (str)
- SEC 3 Coping       → free-text reasoning (str)
- SEC 4 Normative    → free-text reasoning (str)
- Convergence        → structured sentiment classifier (positive / negative / neutral)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


# ============================================================
# REDUCERS
# ============================================================

def add_tokens(current: int, new: int) -> int:
    """Reducer that accumulates token counts across nodes."""
    return current + new


## ============================================================
# REGISTRIES
# ============================================================

class ConvergenceSemEval2017(BaseModel):
    """Sentiment classifier (SemEval-2017 Task 4A) — ternary."""
    sentiment_label: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Clasificación final de sentimiento (taxonomía ternaria SemEval-2017)."
    )
    reasoning: str = Field(
        ..., max_length=1500,
        description="Justificación que conecta las 4 valoraciones SEC con la etiqueta final de sentimiento."
    )


class ConvergenceTweetEval(BaseModel):
    """Emotion classifier (TweetEval Emotion) — 4 classes."""
    sentiment_label: Literal["anger", "joy", "optimism", "sadness"] = Field(
        ...,
        description="Clasificación final de emoción (taxonomía de 4 clases TweetEval)."
    )
    reasoning: str = Field(
        ..., max_length=1500,
        description="Justificación que conecta las 4 valoraciones SEC con la etiqueta final de emoción."
    )


class ConvergenceSemEval2018(BaseModel):
    """Irony classifier (SemEval-2018 Task 3 Subtask A) — binary."""
    sentiment_label: Literal["irony", "not_irony"] = Field(
        ...,
        description="Clasificación final de ironía (taxonomía binaria SemEval-2018 Task 3A)."
    )
    reasoning: str = Field(
        ..., max_length=1500,
        description="Justificación que conecta las 4 valoraciones SEC con la etiqueta final de ironía."
    )


class ConvergenceTASS2019PE(BaseModel):
    """Sentiment classifier (TASS 2019 Perú) — 4 classes."""
    sentiment_label: Literal["P", "N", "NEU", "NONE"] = Field(
        ...,
        description="Clasificación final de sentimiento (taxonomía TASS 2019: P, N, NEU, NONE)."
    )
    reasoning: str = Field(
        ..., max_length=1500,
        description="Justificación que conecta las 4 valoraciones SEC con la etiqueta final de sentimiento."
    )


# ============================================================
# STATE
# ============================================================

@dataclass(kw_only=True)
class Input:
    """Agent input: tweet text and author context."""
    topic: str
    perfil_avatar: str


@dataclass(kw_only=True)
class Output:
    """Agent output: SEC reasonings and final convergence."""
    relevance: Optional[str] = field(default=None)
    implication: Optional[str] = field(default=None)
    coping: Optional[str] = field(default=None)
    normative: Optional[str] = field(default=None)
    convergence: Optional[Dict[str, Any]] = field(default=None)


@dataclass(kw_only=True)
class State(Input):
    """Full internal state for the HERMES agent.

    Each SEC maintains its own message history (isolated context)
    and its own result. Results are shared: each SEC can read
    the outputs of previous SECs.
    """
    # SEC outputs (shared across nodes)
    relevance: Optional[str] = field(default=None)
    implication: Optional[str] = field(default=None)
    coping: Optional[str] = field(default=None)
    normative: Optional[str] = field(default=None)
    convergence: Optional[Dict[str, Any]] = field(default=None)

    # Per-SEC message history (isolated context)
    relevance_messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    implication_messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    coping_messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    normative_messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)

    # Per-SEC loop counters
    relevance_loop: int = field(default=0)
    implication_loop: int = field(default=0)
    coping_loop: int = field(default=0)
    normative_loop: int = field(default=0)

    # Token usage accumulators (each node adds its tokens via reducer)
    total_input_tokens: Annotated[int, add_tokens] = field(default=0)
    total_output_tokens: Annotated[int, add_tokens] = field(default=0)