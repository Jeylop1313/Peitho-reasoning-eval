import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Annotated, Literal, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

# --- 1. EL FORMULARIO DE EVALUACIÓN (SEC 1 - Scherer) ---
# Registro del Perfil Psicológico basado en la subjetividad del Avatar.

class SEC1_Registry(BaseModel):
    """Diagnóstico del Perfil Psicológico: Evaluación de Relevancia."""
    
    # NOVEDAD
    novedad_categoria: Literal["Rutinario", "Inesperado", "Abrupto", "Familiar"] = Field(
        ..., description="Categoría rápida sobre la sorpresa del mensaje."
    )
    novedad_razonamiento: str = Field(
        ..., description="Explicación humana de por qué se siente así para el avatar."
    )
    
    # AGRADO
    agrado_categoria: Literal["Agradable", "Neutral", "Desagradable", "Indignante"] = Field(
        ..., description="Veredicto sobre el tono emocional del lenguaje."
    )
    agrado_razonamiento: str = Field(
        ..., description="Análisis del léxico y la carga emocional desde la perspectiva del sujeto."
    )
    
    # PREDICTIBILIDAD (Corazón del Sarcasmo)
    predictibilidad_categoria: Literal["Lógico", "Incongruente", "Absurdo", "Predecible"] = Field(
        ..., description="¿Tiene sentido lo que dice o rompe la realidad social del avatar?"
    )
    predictibilidad_razonamiento: str = Field(
        ..., description="Choque entre lo dicho y el conocimiento del mundo que tiene el sujeto."
    )
    
    # RELEVANCIA
    relevancia_categoria: Literal["Crítico", "Importante", "Irrelevante", "Tangencial"] = Field(
        ..., description="Impacto del mensaje en la vida o intereses del avatar."
    )
    relevancia_razonamiento: str = Field(
        ..., description="Justificación de por qué este tema le importa (o no) a este perfil específico."
    )
    
    # EL DIAGNÓSTICO
    diagnostico_sarcasmo: str = Field(
        ..., description="Veredicto final: ¿Es sarcasmo? Explica la ironía detectada de forma humana."
    )

# --- 2. CONFIGURACIÓN DE LA MENTE (Estado de Hermes) ---

@dataclass(kw_only=True)
class InputState:
    """La 'Muestra' que entra al microscopio mental."""
    topic: str               # El tuit o comentario
    perfil_avatar: str       # Ej: 'Colombiano de 20 años, universitario'
    
    # El esquema se inyecta automáticamente para que Hermes sepa qué casillas llenar
    extraction_schema: Dict[str, Any] = field(
        default_factory=lambda: SEC1_Registry.model_json_schema()
    )

@dataclass(kw_only=True)
class OutputState:
    """Lo que Hermes nos entrega: El Perfil Psicológico terminado."""
    info: Optional[Dict[str, Any]]

@dataclass(kw_only=True)
class State(InputState):
    """El flujo de conciencia completo del agente."""
    messages: Annotated[List[BaseMessage], add_messages]
    info: Optional[Dict[str, Any]] = field(default=None)
    loop_step: int = field(default=0)