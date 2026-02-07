"""
Hermes: Flujo del Pensamiento Basado en el Perfil Psicológico.
Implementa el ciclo ReAct con el andamiaje cognitivo de Scherer (CPM).
"""

import json
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from enrichment_agent import prompts
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import InputState, OutputState, State, SEC1_Registry 
from enrichment_agent.tools import scrape_website, search
from enrichment_agent.utils import init_model


async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Nodo 'Cerebro': Procesa el tuit bajo el lente del perfil psicológico del avatar.
    """
    configuration = Configuration.from_runnable_config(config)

    # 1. Instanciamos el Formulario SEC 1 como la herramienta 'Info'
    # Esto obliga al LLM a llenar los campos de Scherer (Novedad, Agrado, etc.)
    info_tool = {
        "name": "Info",
        "description": "Llama a esta función para registrar el Perfil Psicológico (SEC 1) completo.",
        "parameters": SEC1_Registry.model_json_schema(), 
    }

    # 2. Inyectamos la 'Muestra': El tuit + el Perfil del Avatar
    # Eliminamos 'metas' para que el modelo use su conocimiento cultural
    p = configuration.prompt.format(
        topic=state.topic,
        perfil=state.perfil_avatar
    )

    messages = [HumanMessage(content=p)] + state.messages

    raw_model = init_model(config)
    # El agente puede investigar en web si no entiende un modismo, pero su meta es el formulario.
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))

    info = None
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    
    if info is not None:
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:
        response_messages.append(
            HumanMessage(content="Por favor, completa el Registro de Appraisal llamando a la herramienta 'Info'.")
        )

    return {
        "messages": response_messages,
        "info": info,
        # Incrementamos el paso acumulativamente para la 'Economía de Procesamiento'
        "loop_step": state.loop_step + 1, 
    }


class PerfilIsSatisfactory(BaseModel):
    """Valida la coherencia pragmática del análisis psicológico."""
    reason: List[str] = Field(
        description="Razones por las que el análisis es fiel (o no) al avatar y a la teoría CPM."
    )
    is_satisfactory: bool = Field(
        description="¿Es el diagnóstico psicológico coherente y estable?"
    )
    improvement_instructions: Optional[str] = Field(
        description="Instrucciones para ajustar el análisis si hay inconsistencias pragmáticas.",
        default=None,
    )


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Nodo 'Crítico Interno': Evalúa si el diagnóstico de sarcasmo suena 'humano' y lógico.
    """
    avatar = state.perfil_avatar
    last_message = state.messages[-1]
    presumed_info = state.info

    # El Crítico ahora evalúa coherencia general sin depender de una lista de metas fija
    checker_prompt = """Eres el Crítico Interno de Hermes. \
Estás revisando un Perfil Psicológico para el tuit: "{topic}".
El observador es: {avatar}.

¿Es coherente este análisis de sarcasmo? 
Revisa si la Predictibilidad y la Relevancia tienen sentido para este perfil cultural.
Si el análisis es genérico o 'parece robot', marca insatisfactorio.

Análisis propuesto:
{presumed_info}"""

    p1 = checker_prompt.format(
        topic=state.topic,
        avatar=avatar,
        presumed_info=json.dumps(presumed_info or {}, indent=2)
    )
    
    messages = [HumanMessage(content=p1)]
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(PerfilIsSatisfactory)
    response = cast(PerfilIsSatisfactory, await bound_model.ainvoke(messages))

    if response.is_satisfactory and presumed_info:
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.reason),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Análisis inconsistente:\n{response.improvement_instructions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }

# --- Lógica de Rutas y Grafo ---

def route_after_agent(state: State) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    return "tools"

def route_after_checker(state: State, config: RunnableConfig) -> Literal["__end__", "call_agent_model"]:
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]

    # Implementamos el Gatekeeper de Cierre
    if state.loop_step < configurable.max_loops:
        if not state["info"]:
            return "call_agent_model"
        if isinstance(last_message, ToolMessage) and last_message.status == "error":
            return "call_agent_model"
        return "__end__"
    return "__end__"

# Creación del Grafo
workflow = StateGraph(State, config_schema=Configuration)
workflow.add_node(call_agent_model)
workflow.add_node(reflect)
workflow.add_node("tools", ToolNode([search, scrape_website]))

workflow.add_edge("__start__", "call_agent_model")
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_edge("tools", "call_agent_model")
workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "Hermes_Psychological_Profile"