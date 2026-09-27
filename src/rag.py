"""Retriever + cadena LCEL grounded + get_rag_response asíncrona."""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import ANTHROPIC_MODEL, COLLECTION_NAME, PERSIST_DIR, TOP_K, get_embeddings
from src.schemas import RAGResponse

SYSTEM_PROMPT = """Eres un asistente técnico de FinCrece S.A., una financiera de créditos personales.

Respondé ÚNICAMENTE en base al CONTEXTO provisto a continuación. No uses conocimiento propio, \
no supongas información que no esté explícita en el CONTEXTO y no completes datos faltantes.

- Si la respuesta está en el CONTEXTO: marcá "encontrado" en true, respondé de forma precisa \
y completá "referencias" con al menos un elemento, citando el archivo fuente ("fuente") y una \
cita breve del texto usado ("fragmento").
- Si la respuesta NO está en el CONTEXTO: respondé exactamente "No lo sé" en el campo \
"respuesta", marcá "encontrado" en false y dejá "referencias" como una lista vacía.

CONTEXTO:
{context}

{format_instructions}
"""

_parser = PydanticOutputParser(pydantic_object=RAGResponse)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[Fuente: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def get_chain():
    llm = ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0)
    retriever = get_retriever()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt.partial(format_instructions=_parser.get_format_instructions())
        | llm
        | _parser
    )
    # Resiliencia: reintenta si el LLM devuelve un JSON mal formado / que no parsea.
    return chain.with_retry(stop_after_attempt=3)


async def get_rag_response(query: str) -> RAGResponse:
    """Búsqueda de similitud + prompt grounded + llamada asíncrona al LLM + parseo Pydantic."""
    chain = get_chain()
    return await chain.ainvoke(query)


def get_relevant_with_scores(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Fragmentos recuperados junto con su distancia, para inspección/depuración."""
    return get_vectorstore().similarity_search_with_score(query, k=k)
