import asyncio
import csv
import os
import time
from dotenv import load_dotenv
from src.enrichment_agent.graph import graph
from src.enrichment_agent.state import DatosClinica 

load_dotenv()

# --- NOMBRES DE ARCHIVOS ---
INPUT_FILE = "clinicasmedellin_input.csv"       # Tu lista de 100 clínicas
OUTPUT_FILE = "resultado_ventas.csv"    # Aquí se guardarán los hallazgos

def guardar_fila(datos):
    """Guarda los datos encontrados en el archivo de resultados."""
    archivo_existe = os.path.isfile(OUTPUT_FILE)
    
    # Estas columnas deben coincidir con tu state.py
    columnas = [
        "nombre_clinica", "icp", "nombre_ceo", "email_ceo", 
        "telefono_ceo", "website", "descripcion"
    ]
    
    # 'utf-8-sig' es para que Excel lea bien las tildes y ñ
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columnas, extrasaction='ignore')
        if not archivo_existe:
            writer.writeheader()
        writer.writerow(datos)

async def procesar_lote():
    # 1. Verificar que subiste el archivo
    if not os.path.isfile(INPUT_FILE):
        print(f"❌ ERROR: No encuentro '{INPUT_FILE}' en la carpeta data-enrichment.")
        return

    # 2. Leer la lista de clínicas
    lista_clinicas = []
    with open(INPUT_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Asegurarse de leer la columna correcta 'nombre_ips'
            if row.get("nombre_ips"):
                lista_clinicas.append(row["nombre_ips"].strip())

    total = len(lista_clinicas)
    print(f"🚀 INICIANDO PROCESO MASIVO: {total} clínicas detectadas.\n")

    # 3. Bucle de trabajo (Una por una)
    for i, nombre_clinica in enumerate(lista_clinicas, 1):
        print(f"[{i}/{total}] 🕵️  Investigando: {nombre_clinica}...")
        
        try:
            # Llamamos al Agente
            resultado = await graph.ainvoke({
                "topic": nombre_clinica,
                "extraction_schema": DatosClinica.model_json_schema()
            })
            
            info = resultado.get("info")
            
            if info:
                guardar_fila(info)
                print(f"   ✅ Clasificada como: {info.get('icp', 'N/A')}")
            else:
                print("   ⚠️  No se encontró info.")
                guardar_fila({"nombre_clinica": nombre_clinica, "descripcion": "NO ENCONTRADO"})

        except Exception as e:
            print(f"   ❌ Error técnico: {e}")
            guardar_fila({"nombre_clinica": nombre_clinica, "descripcion": f"ERROR: {e}"})
        
        print("-" * 30)

    print(f"\n🏁 ¡TERMINADO! Abre el archivo '{OUTPUT_FILE}' en Excel.")

if __name__ == "__main__":
    asyncio.run(procesar_lote())