"""Ejecuta la ingesta (si hace falta) y dos pruebas del sistema RAG:
una pregunta con respuesta en los documentos y una "pregunta trampa" sin respuesta.
"""

import asyncio

from src.ingest import run_ingest
from src.rag import get_rag_response, get_relevant_with_scores

PREGUNTA_VALIDA = "¿A partir de cuántos días de atraso se deriva la deuda a cobranza judicial?"
PREGUNTA_TRAMPA = "¿Cuáles son los requisitos para acceder a un crédito hipotecario?"


def print_fragmentos(query: str) -> None:
    print(f"\nFragmentos recuperados para: {query!r}")
    for doc, distancia in get_relevant_with_scores(query):
        fuente = doc.metadata.get("source", "?")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"  distancia={distancia:.4f} | fuente={fuente} | {preview}...")


async def run_case(etiqueta: str, query: str) -> None:
    print(f"\n{'=' * 70}\n{etiqueta}\n{'=' * 70}")
    print(f"Pregunta: {query}")
    print_fragmentos(query)

    respuesta = await get_rag_response(query)

    print(f"\nRespuesta:   {respuesta.respuesta}")
    print(f"Encontrado:  {respuesta.encontrado}")
    if respuesta.referencias:
        print("Referencias:")
        for ref in respuesta.referencias:
            print(f"  - {ref.fuente}: {ref.fragmento}")
    else:
        print("Referencias: (ninguna)")


async def main() -> None:
    await run_case("Prueba 1 — pregunta con respuesta en los documentos", PREGUNTA_VALIDA)
    await run_case("Prueba 2 — pregunta trampa (sin respuesta en los documentos)", PREGUNTA_TRAMPA)

    # Bonus: ambas consultas en paralelo con asyncio.gather.
    print(f"\n{'=' * 70}\nBonus — ambas consultas en paralelo (asyncio.gather)\n{'=' * 70}")
    resultados = await asyncio.gather(
        get_rag_response(PREGUNTA_VALIDA),
        get_rag_response(PREGUNTA_TRAMPA),
    )
    for query, resultado in zip([PREGUNTA_VALIDA, PREGUNTA_TRAMPA], resultados):
        print(f"[{resultado.encontrado}] {query}")


if __name__ == "__main__":
    run_ingest()
    asyncio.run(main())
