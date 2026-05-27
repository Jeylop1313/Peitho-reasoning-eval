"""
HERMES — Cognitive Appraisal Agent (Scherer CPM)
=================================================
Architecture: 4 sequential SEC nodes with explicit reasoning,
followed by a convergence node that produces the final sentiment label.

Flow:
  relevance ⇄ tools → implication → coping → normative → convergence → END
"""

from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from enrichment_agent.State import (
    State,
    ConvergenceSemEval2017,
    ConvergenceTweetEval,
    ConvergenceSemEval2018,
    ConvergenceTASS2019PE,
)
from enrichment_agent.relevance import RELEVANCE_PROMPT
from enrichment_agent.implications import IMPLICATIONS_PROMPT
from enrichment_agent.coping import COPING_PROMPT
from enrichment_agent.normative import NORMATIVE_PROMPT
from enrichment_agent.converge_semeval2017 import CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_SEMEVAL2017
from enrichment_agent.converge_tweeteval import CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_TWEETEVAL
from enrichment_agent.converge_semeval2018 import CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_SEMEVAL2018
from enrichment_agent.converge_tass2019PE import CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_TASS2019PE

from enrichment_agent.configuration import Configuration
from enrichment_agent.tools import search_news, search
from enrichment_agent.utils import init_model


# ============================================================
# CONVERGENCE VARIANTS (one per dataset)
# ============================================================
CONVERGENCE_VARIANTS = {
    "semeval2017": (CONVERGENCE_PROMPT_SEMEVAL2017, ConvergenceSemEval2017),
    "tweeteval":   (CONVERGENCE_PROMPT_TWEETEVAL,   ConvergenceTweetEval),
    "semeval2018": (CONVERGENCE_PROMPT_SEMEVAL2018, ConvergenceSemEval2018),
    "tass2019PE":  (CONVERGENCE_PROMPT_TASS2019PE,  ConvergenceTASS2019PE),
}

DEFAULT_CONVERGENCE_VARIANT = "semeval2017"


# ============================================================
# TOOLS
# ============================================================
USE_NEWS_SEARCH = False  # True for real-time datasets, False for SemEval historical
TOOLS = [search, search_news] if USE_NEWS_SEARCH else [search]
TOOL_NAMES = {t.name for t in TOOLS}


# ============================================================
# TOKEN EXTRACTION HELPER
# ============================================================

def _extract_tokens(response: AIMessage) -> tuple[int, int]:
    """Extract input/output token counts from a Groq response."""
    usage = response.response_metadata.get("token_usage", {})
    return (
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


# ============================================================
# SEC NODE FACTORY
# ============================================================

def _build_sec_node(
    prompt_template: str,
    messages_key: str,
    result_key: str,
    loop_key: str,
    external_tools: Optional[list] = None,
):
    """Build a SEC node with a two-phase loop.

    Phase 0 (exploratory reasoning): the model reasons freely and may
    invoke search tools if *external_tools* is provided.
    Phase 1+ (forced conclusion): the model must commit to a text output
    without tool access.
    """
    has_tools = external_tools is not None

    async def sec_node(
        state: State, *, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:

        p = prompt_template.format(
            comment=state.topic,
            Avatar=state.perfil_avatar,
            relevance_output=state.relevance or "",
            implication_output=state.implication or "",
            coping_output=state.coping or "",
        )

        current_messages = getattr(state, messages_key, [])
        current_loop = getattr(state, loop_key, 0)

        messages = [HumanMessage(content=p)] + list(current_messages)

        raw_model = init_model(config)

        # ── PHASE 0: EXPLORATORY REASONING ─────────────────────
        if current_loop == 0:
            messages.append(
                HumanMessage(
                    content=(
                        "FASE DE RAZONAMIENTO.\n"
                        + (
                            "ANTES de razonar: ¿tienes suficiente contexto sobre a qué se refiere el tweet?\n"
                            "Si NO: usa la herramienta de búsqueda disponible y razona con los resultados.\n"
                            "Si SÍ: razona directamente.\n\n"
                            if has_tools else ""
                        )
                        + "RAZONA sobre lo que significa el comentario para este SEC:\n"
                        "- ¿Qué evidencia es relevante?\n"
                        "- ¿Qué implica para el avatar?\n"
                        "NO registres tu evaluación todavía."
                    )
                )
            )
            if has_tools:
                model = raw_model.bind_tools(external_tools, tool_choice="auto")
            else:
                model = raw_model

        # ── PHASE 1+: FORCED CONCLUSION ────────────────────────
        else:
            messages.append(
                HumanMessage(
                    content=(
                        "RAZONA sobre lo que significa el comentario para este SEC:\n"
                        "- ¿Qué evidencia es relevante?\n"
                        "- ¿Qué implica para el avatar?\n"
                        "NO registres tu evaluación todavía."
                    )
                )
            )
            model = raw_model

        response = cast(AIMessage, await model.ainvoke(messages))

        info = response.content if isinstance(response.content, str) and response.content else None
        if info:
            print(f"[{result_key} loop {current_loop}]: {info[:120]}...")

        input_tok, output_tok = _extract_tokens(response)

        return {
            messages_key: [response],
            result_key: None if response.tool_calls else info,
            loop_key: current_loop + 1,
            "total_input_tokens": input_tok,
            "total_output_tokens": output_tok,
        }

    return sec_node


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_relevance(
    state: State,
) -> Literal["relevance_tools", "implication_node", "relevance_node"]:
    if state.relevance is not None:
        return "implication_node"
    messages = state.relevance_messages
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if any(tc["name"] in TOOL_NAMES for tc in last_msg.tool_calls):
                return "relevance_tools"
    return "relevance_node"


def route_after_implication(
    state: State,
) -> Literal["coping_node", "implication_node"]:
    if state.implication is not None:
        return "coping_node"
    return "implication_node"


def route_after_coping(
    state: State,
) -> Literal["normative_node", "coping_node"]:
    if state.coping is not None:
        return "normative_node"
    return "coping_node"


def route_after_normative(
    state: State,
) -> Literal["convergence_node", "normative_node"]:
    if state.normative is not None:
        return "convergence_node"
    return "normative_node"


async def convergence_node(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synthesize the 4 SEC outputs into a final sentiment label.

    Selects prompt + schema according to `convergence_variant` in the
    RunnableConfig. Falls back to DEFAULT_CONVERGENCE_VARIANT if absent.
    """

    # Read variant from config (default: semeval2017)
    variant = DEFAULT_CONVERGENCE_VARIANT
    if config is not None:
        variant = config.get("configurable", {}).get(
            "convergence_variant", DEFAULT_CONVERGENCE_VARIANT
        )

    if variant not in CONVERGENCE_VARIANTS:
        raise ValueError(
            f"Unknown convergence_variant '{variant}'. "
            f"Valid options: {list(CONVERGENCE_VARIANTS.keys())}"
        )

    convergence_prompt, convergence_schema = CONVERGENCE_VARIANTS[variant]

    convergence_tool = {
        "name": "ConvergenceInfo",
        "description": "Registra tu síntesis integradora de los 4 SEC.",
        "parameters": convergence_schema.model_json_schema(),
    }

    p = convergence_prompt.format(
        comment=state.topic,
        Avatar=state.perfil_avatar,
        relevance=(state.relevance or "").replace("{", "{{").replace("}", "}}"),
        implication=(state.implication or "").replace("{", "{{").replace("}", "}}"),
        coping=(state.coping or "").replace("{", "{{").replace("}", "}}"),
        normative=(state.normative or "").replace("{", "{{").replace("}", "}}"),
    )

    raw_model = init_model(config)
    model = raw_model.bind_tools([convergence_tool], tool_choice="ConvergenceInfo")

    response = cast(AIMessage, await model.ainvoke([HumanMessage(content=p)]))

    result = None
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "ConvergenceInfo":
                result = tool_call["args"]
                print(f"Convergence completed (variant: {variant})")
                break

    input_tok, output_tok = _extract_tokens(response)

    return {
        "convergence": result,
        "total_input_tokens": input_tok,
        "total_output_tokens": output_tok,
    }


# ============================================================
# NODE INSTANTIATION
# ============================================================

# SEC 1: Relevance (with search tools)
relevance_node = _build_sec_node(
    RELEVANCE_PROMPT,
    "relevance_messages", "relevance", "relevance_loop",
    external_tools=TOOLS,
)
relevance_node.__name__ = "relevance_node"

# SEC 2: Implication (reasoning only)
implication_node = _build_sec_node(
    IMPLICATIONS_PROMPT,
    "implication_messages", "implication", "implication_loop",
)
implication_node.__name__ = "implication_node"

# SEC 3: Coping Potential (reasoning only)
coping_node = _build_sec_node(
    COPING_PROMPT,
    "coping_messages", "coping", "coping_loop",
)
coping_node.__name__ = "coping_node"

# SEC 4: Normative Significance (reasoning only)
normative_node = _build_sec_node(
    NORMATIVE_PROMPT,
    "normative_messages", "normative", "normative_loop",
)
normative_node.__name__ = "normative_node"

# Tool node (Relevance only)
relevance_tools = ToolNode(TOOLS, messages_key="relevance_messages")


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(State, config_schema=Configuration)

workflow.add_node("relevance_node", relevance_node)
workflow.add_node("relevance_tools", relevance_tools)
workflow.add_node("implication_node", implication_node)
workflow.add_node("coping_node", coping_node)
workflow.add_node("normative_node", normative_node)
workflow.add_node("convergence_node", convergence_node)

workflow.add_edge("__start__", "relevance_node")

workflow.add_conditional_edges("relevance_node", route_after_relevance)
workflow.add_edge("relevance_tools", "relevance_node")
workflow.add_conditional_edges("implication_node", route_after_implication)
workflow.add_conditional_edges("coping_node", route_after_coping)
workflow.add_conditional_edges("normative_node", route_after_normative)
workflow.add_edge("convergence_node", "__end__")

graph = workflow.compile()
graph.name = "Hermes"

print("Graph compiled: Relevance -> Implication -> Coping -> Normative -> Convergence -> END")