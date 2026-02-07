"""Los parámetros técnicos del Agente."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from enrichment_agent import prompts

@dataclass(kw_only=True)
class Configuration:
    """Configuración técnica para la calibración del agente."""

    # --- Selección de Modelo ---
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gemini-2.5-flash-lite", 
        metadata={
            "description": "El modelo de Gemini a usar. Se recomienda gemini-1.5-flash para velocidad y costo."
        },
    )

    model_provider: str = field(
        default="google_genai",
        metadata={
            "description": "El proveedor del modelo. Usar 'google_genai' para Google AI Studio (gratuito)."
        },
    )

    # --- El Lente Teórico (El Prompt) ---
    # Apunta al prompt que operacionaliza el SEC 1 de Scherer.
    prompt: str = field(
        default=prompts.MAIN_PROMPT,
        metadata={
            "description": "Plantilla que guía los pasos del SEC 1 (Novedad, Agrado, Metas)."
        },
    )

    # --- Límites de Herramientas ---
    max_search_results: int = field(
        default=3,
        metadata={
            "description": "Máximo de evidencias externas para validar la Predictibilidad."
        },
    )

    # --- Robustez vs. Economía de Procesamiento ---
    max_loops: int = field(
        default=4,
        metadata={
            "description": "Límite de intentos para alcanzar el Cierre (Closure) en el peritaje."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Carga la configuración técnica para la ejecución actual."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})