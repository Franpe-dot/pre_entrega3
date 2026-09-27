"""Configuración central del sistema RAG.

Fuente única de verdad para rutas, nombre de colección y modelo de embeddings.
Tanto `ingest.py` (indexación) como `rag.py` (consulta) importan `get_embeddings()`
desde este módulo para garantizar que se usa siempre el mismo modelo: si el modelo de
embeddings de la ingesta no coincide con el de la consulta, la distancia vectorial
deja de tener sentido (ver README, sección "Decisiones técnicas").
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = str(BASE_DIR / "vectorstore")
COLLECTION_NAME = "politicas_credito"

# intfloat/multilingual-e5-base: 512 tokens de contexto (soporta chunks de 500 tokens),
# multilingüe (el dataset está en español) y liviano en comparación con BAAI/bge-m3.
# Requiere prefijar "query: " / "passage: " a los textos antes de codificarlos
# (ver clase E5Embeddings más abajo).
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# Chunking: mínimo exigido por la consigna (500 tokens, medidos con tiktoken, 50 de overlap).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# top_k entre 3 y 5 para evitar el problema de "contexto infinito" / lost in the middle.
TOP_K = 4

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


class E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings con los prefijos asimétricos que exige la familia e5.

    e5 fue entrenado para distinguir "pasajes" de "consultas": sin este prefijo el
    espacio vectorial no es el que el modelo espera y la similitud pierde precisión.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


@lru_cache(maxsize=1)
def get_embeddings() -> E5Embeddings:
    return E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
