# RAG local — Políticas de crédito de FinCrece S.A. (financiera ficticia)

Sistema de recuperación semántica (RAG) end-to-end: indexa documentos `.md` en ChromaDB,
recupera los fragmentos más relevantes para una pregunta y genera una respuesta **grounded**
(solo con esa información) usando un LLM de Anthropic vía LangChain (LCEL).

Pre-entrega 3 — Módulo 3 (embeddings, chunking y ChromaDB).

## Arquitectura

```
 /data (*.md)
     │  limpieza + chunking (RecursiveCharacterTextSplitter, 500 tokens / 50 overlap)
     ▼
 src/ingest.py ──── embeddings (intfloat/multilingual-e5-base) ────▶  ChromaDB persistente
                                                                        (./vectorstore)
                                                                              │
 pregunta del usuario                                                        │ similarity_search
     │                                                                       ▼
     └──▶ src/rag.py: retriever (top_k=4) ──▶ prompt "filtro de veracidad" ──▶ Claude (Anthropic)
                                                                              │
                                                              PydanticOutputParser
                                                                              ▼
                                                        RAGResponse(respuesta, encontrado, referencias)
```

- **Ingesta** ([src/ingest.py](src/ingest.py)): lee `.md`/`.txt` de `/data`, limpia el texto,
  fragmenta con `RecursiveCharacterTextSplitter.from_tiktoken_encoder` y hace `upsert` en
  ChromaDB con IDs deterministas (`archivo::índice_de_chunk`), verificando antes si la
  colección ya existe y tiene documentos.
- **Retriever + generación** ([src/rag.py](src/rag.py)): arma una cadena LCEL
  `{context, question} | prompt | llm | PydanticOutputParser`, donde el prompt instruye al
  modelo a responder únicamente con el contexto recuperado y a decir **"No lo sé"** si la
  respuesta no está allí.
- **Esquemas** ([src/schemas.py](src/schemas.py)): `RAGResponse` (respuesta, `encontrado`,
  `referencias`), con un validador que exige que `referencias` esté vacía si `encontrado=False`
  y no vacía si `encontrado=True`.
- **Configuración** ([src/config.py](src/config.py)): única fuente de verdad para el modelo
  de embeddings, `top_k`, tamaño de chunk y rutas — la importan tanto la ingesta como el
  retriever, para evitar el error de usar modelos de embeddings distintos en cada lado.

## Instalación

Requiere Python 3.10+ (probado con Python 3.14).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Copiá `.env.example` a `.env` y completá tu API key de Anthropic:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

El modelo de embeddings (`intfloat/multilingual-e5-base`) es local y se descarga
automáticamente de HuggingFace la primera vez que se ejecuta la ingesta (no requiere API key).

## Ejecución

**1. Indexar los documentos** (crea `./vectorstore`; si ya existe y tiene datos, se omite):

```bash
python -m src.ingest
```

Para forzar una re-indexación completa (por ejemplo, si cambiaron los `.md` o el modelo de
embeddings):

```bash
python -m src.ingest --reindex
```

**2. Correr las pruebas end-to-end** (indexa si hace falta y ejecuta las dos preguntas de
prueba + una ejecución concurrente con `asyncio.gather`):

```bash
python main.py
```

`main.py` llama a `await get_rag_response(query)` ([src/rag.py](src/rag.py)), la función
asíncrona pedida por la consigna, que: busca por similitud en ChromaDB → arma el prompt con
los fragmentos recuperados → llama a Claude de forma asíncrona (`chain.ainvoke`) → parsea la
respuesta a `RAGResponse` con `PydanticOutputParser`.

## Decisiones técnicas

- **LLM:** Anthropic Claude (`langchain-anthropic`), configurable por `ANTHROPIC_MODEL`
  (default `claude-haiku-4-5-20251001`).
- **Embeddings:** `intfloat/multilingual-e5-base` (local, `sentence-transformers` vía
  `langchain-huggingface`).
  - Es **multilingüe** (el dataset está en español) y soporta hasta **512 tokens** de
    contexto, suficiente para los chunks de 500 tokens exigidos por la consigna —
    `paraphrase-multilingual-MiniLM-L12-v2`, por ejemplo, trunca a 128 tokens y hubiera
    cortado en silencio buena parte de cada chunk.
  - e5 requiere prefijar los textos con `"query: "` (consultas) y `"passage: "`
    (documentos) para funcionar correctamente; esto se implementa en la clase
    `E5Embeddings` de [src/config.py](src/config.py), que envuelve `HuggingFaceEmbeddings`.
  - El modelo se define **una sola vez** en `config.get_embeddings()` y lo importan tanto
    `ingest.py` como `rag.py`, para evitar el error #1 (embeddings no coincidentes entre
    indexación y consulta).
- **Chunking:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (encoding
  `cl100k_base`), `chunk_size=500`, `chunk_overlap=50`, separadores
  `["\n\n", "\n", " ", ""]`. Se mide en tokens (no caracteres) para evitar truncamientos
  silenciosos del modelo de embeddings.
- **top_k = 4:** dentro del rango sugerido (3–5) para evitar el problema de "contexto
  infinito" / *lost in the middle*.
- **Distancia coseno:** la colección de Chroma se crea con `metadata={"hnsw:space": "cosine"}`
  en vez del L2 por defecto.
- **Persistencia:** `chromadb.PersistentClient(path="./vectorstore")`. `ingest.py` chequea
  si la colección ya existe y tiene documentos antes de volver a generar embeddings; `--reindex`
  fuerza el borrado y la re-indexación completa.
- **IDs deterministas:** `f"{nombre_de_archivo}::{índice_de_chunk}"`, usados con
  `collection.upsert(...)` para que reindexar no genere duplicados.
- **Resiliencia:** la cadena LCEL se envuelve con `.with_retry(stop_after_attempt=3)` para
  reintentar automáticamente si el LLM devuelve una respuesta que no puede parsearse como
  `RAGResponse`.
- **Prompt "filtro de veracidad":** instruye a responder solo con el CONTEXTO, sin usar
  conocimiento previo, y a devolver exactamente `"No lo sé"` con `encontrado=false` y
  `referencias=[]` cuando la respuesta no está en el contexto.

## Dataset de ejemplo

`/data` contiene 4 documentos `.md` con las políticas (ficticias) de **FinCrece S.A.**, una
financiera que otorga **solo créditos personales** (no hipotecarios ni prendarios):

| Archivo | Contenido |
|---|---|
| [`01_otorgamiento.md`](data/01_otorgamiento.md) | Requisitos, montos, plazos y proceso de aprobación |
| [`02_tasas_y_costos.md`](data/02_tasas_y_costos.md) | TNA, CFTEA, comisiones y forma de pago |
| [`03_mora_y_cobranzas.md`](data/03_mora_y_cobranzas.md) | Días de gracia, recargos y derivación a cobranza judicial |
| [`04_refinanciacion_cancelacion.md`](data/04_refinanciacion_cancelacion.md) | Refinanciación de deudas en mora y cancelación anticipada |

## Pruebas (`main.py`)

1. **Pregunta válida:** *"¿A partir de cuántos días de atraso se deriva la deuda a cobranza
   judicial?"* — la respuesta está en `03_mora_y_cobranzas.md` (día 91). Se espera
   `encontrado=True` y una referencia a ese archivo.
2. **Pregunta trampa:** *"¿Cuáles son los requisitos para acceder a un crédito hipotecario?"*
   — FinCrece S.A. solo otorga créditos personales, por lo que el dato no existe en los
   documentos (aunque el retriever igual trae fragmentos parecidos sobre requisitos de
   crédito personal, para poner a prueba el *grounding* del LLM y no solo al retriever). Se
   espera `respuesta="No lo sé"`, `encontrado=False` y `referencias=[]`.

`main.py` imprime, para cada pregunta, los fragmentos recuperados con su distancia coseno
antes de mostrar la respuesta final, para poder inspeccionar qué contexto recibió el modelo.

### Ejemplo de salida esperada

```
======================================================================
Prueba 1 — pregunta con respuesta en los documentos
======================================================================
Pregunta: ¿A partir de cuántos días de atraso se deriva la deuda a cobranza judicial?

Fragmentos recuperados para: '...'
  distancia=0.1137 | fuente=03_mora_y_cobranzas.md | # Política de Mora y Cobranzas...
  distancia=0.1384 | fuente=03_mora_y_cobranzas.md | ## Recargos por mora...
  ...

Respuesta:   A partir del día 91 de atraso, si la deuda no fue regularizada ni
             refinanciada, el caso se deriva a cobranza judicial.
Encontrado:  True
Referencias:
  - 03_mora_y_cobranzas.md: "A partir del día 91 de atraso [...] se deriva a cobranza judicial"

======================================================================
Prueba 2 — pregunta trampa (sin respuesta en los documentos)
======================================================================
Pregunta: ¿Cuáles son los requisitos para acceder a un crédito hipotecario?
...
Respuesta:   No lo sé
Encontrado:  False
Referencias: (ninguna)
```

*(La salida real depende del modelo de Anthropic usado; el `respuesta` exacto puede variar
en su redacción, pero `encontrado` y `referencias` deben cumplir lo anterior.)*

## Estructura del repositorio

```
pre_entrega3/
├── data/                       # 4 .md de políticas de crédito ficticias
├── src/
│   ├── config.py                # rutas, colección, modelo de embeddings, top_k
│   ├── ingest.py                # limpieza + chunking + upsert en Chroma
│   ├── schemas.py                # modelos Pydantic de la respuesta
│   └── rag.py                    # retriever + cadena LCEL + get_rag_response async
├── main.py                       # corre las pruebas (pregunta válida + trampa)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Seguridad

No hay API keys en el repositorio: se leen desde `.env` (ignorado por git) mediante
`python-dotenv`. Usar `.env.example` como plantilla.
