"""Módulo de ingesta: lee /data, fragmenta (chunking) y persiste en ChromaDB.

Verifica si la colección ya existe y tiene documentos antes de volver a indexar
(usar --reindex para forzar una re-indexación completa).

Uso:
    python -m src.ingest             # indexa solo si la colección está vacía
    python -m src.ingest --reindex   # fuerza la re-indexación completa
"""

import argparse
import re
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR, PERSIST_DIR, get_embeddings


def clean_text(text: str) -> str:
    """Limpieza previa: normaliza saltos de línea y espacios, conserva \n\n como
    límite de párrafo (el splitter jerárquico lo usa como primer separador)."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    documents = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        documents.append((path.name, clean_text(path.read_text(encoding="utf-8"))))
    return documents


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_id(source: str, index: int) -> str:
    """ID determinista: re-indexar el mismo documento pisa (upsert) los mismos IDs
    en lugar de generar duplicados."""
    return f"{source}::{index}"


def collection_is_populated(client: chromadb.ClientAPI) -> bool:
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return False
    return collection.count() > 0


def run_ingest(reindex: bool = False) -> None:
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    if not reindex and collection_is_populated(client):
        print(
            f"La colección '{COLLECTION_NAME}' ya existe y tiene documentos: "
            "se omite la ingesta (usa --reindex para forzarla)."
        )
        return

    if reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # hnsw:space=cosine: ChromaDB usa L2 por defecto, pero la convención en
    # embeddings semánticos (y la que espera e5) es similitud coseno.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"No se encontraron archivos .md/.txt en {DATA_DIR}")

    splitter = build_splitter()
    embeddings = get_embeddings()

    ids, texts, metadatas = [], [], []
    for source, text in documents:
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(chunk_id(source, i))
            texts.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

    print(f"Indexando {len(texts)} fragmentos de {len(documents)} documento(s)...")
    vectors = embeddings.embed_documents(texts)
    collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    print(f"Ingesta completa: {len(texts)} fragmentos en '{COLLECTION_NAME}' ({PERSIST_DIR}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en ChromaDB.")
    parser.add_argument("--reindex", action="store_true", help="Fuerza la re-indexación completa.")
    args = parser.parse_args()
    run_ingest(reindex=args.reindex)
