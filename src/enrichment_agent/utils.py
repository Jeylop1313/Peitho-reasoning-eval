"""Utility functions used in our graph."""

from typing import Optional

# CAMBIO 1: Importamos la librería de Google Generative AI (Gratuita)
# en lugar de la función genérica init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from enrichment_agent.configuration import Configuration


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the configured chat model."""
    configuration = Configuration.from_runnable_config(config)
    
    # CAMBIO 2: En lugar de dejar que init_chat_model decida (y falle),
    # instanciamos directamente el modelo gratuito.
    # Esto buscará automáticamente la variable GOOGLE_API_KEY en tu entorno.
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Puedes cambiarlo a "gemini-1.5-pro" si prefieres
        temperature=0
    )
