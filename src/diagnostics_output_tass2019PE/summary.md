# HERMES — Error Diagnostics TASS 2019 PE (Full Pipeline (4 SECs))

**File:** `results_tass2019PE.json`  
**Format:** Full Pipeline (4 SECs)  
**Classes:** P, N, NEU, NONE  
**Total evaluated:** 964 (skipped: 2)  
**Correct:** 417  
**Errors:** 547  

## Métricas

| Métrica | Valor |
|---------|-------|
| AvgRec | 0.4778 |
| Macro-F1 | 0.3406 |
| F1^PN | 0.6028 |
| Accuracy | 0.4326 |
| Micro-F1 | 0.4326 |

## Métricas por Clase

| Clase | Precisión | Recall | F1 | Gold | Pred |
|-------|-----------|--------|----|------|------|
| P | 0.4118 | 0.9074 | 0.5665 | 216 | 476 |
| N | 0.5000 | 0.8855 | 0.6391 | 227 | 402 |
| NEU | 0.2326 | 0.1183 | 0.1569 | 169 | 86 |
| NONE | 0.0000 | 0.0000 | 0.0000 | 352 | 0 |

## Matriz de Confusión

| Gold \ Pred | Positivo | Negativo | Neutral | Sin sentimiento |
|------------|-------:|-------:|-------:|-------:|
| P | 196 | 15 | 5 | 0 |
| N | 17 | 201 | 9 | 0 |
| NEU | 80 | 69 | 20 | 0 |
| NONE | 183 | 117 | 52 | 0 |

## Distribución de Errores

| Patrón | Count | % del Total | Archivo |
|--------|-------|------------:|---------|
| NONE>P | 183 | 33.5% | `errors_NONE_to_P.md` |
| NONE>N | 117 | 21.4% | `errors_NONE_to_N.md` |
| NEU>P | 80 | 14.6% | `errors_NEU_to_P.md` |
| NEU>N | 69 | 12.6% | `errors_NEU_to_N.md` |
| NONE>NEU | 52 | 9.5% | `errors_NONE_to_NEU.md` |
| N>P | 17 | 3.1% | `errors_N_to_P.md` |
| P>N | 15 | 2.7% | `errors_P_to_N.md` |
| N>NEU | 9 | 1.6% | `errors_N_to_NEU.md` |
| P>NEU | 5 | 0.9% | `errors_P_to_NEU.md` |

## Uso de Recursos

| Métrica | Total | Promedio por Tweet |
|---------|------:|-------------------:|
| Input tokens | 9,757,363 | 10,122 |
| Output tokens | 1,886,085 | 1,957 |
| Total tokens | 11,643,448 | 12,078 |
| Tiempo (segundos) | 8,104.6 | 8.4 |
