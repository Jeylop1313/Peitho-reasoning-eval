# 1. Imagen base
FROM python:3.11-slim

# 2. Instalamos uv
RUN pip install uv

# 3. Directorio de trabajo
WORKDIR /app

# 4. Copiamos TODO el contenido primero
# Esto incluye src/, README.md, pyproject.toml y uv.lock
COPY . .

# 5. Ahora sí, instalamos las dependencias
# Al haber copiado todo, uv encontrará la carpeta src/enrichment_agent
RUN uv pip install --system --no-cache .

# 6. Exponer el puerto (opcional para LangGraph)
EXPOSE 8000

# 7. Ejecutar el agente
CMD ["python", "prueba.py"]