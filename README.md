# RAG PDF QA System

A public-ready local PDF RAG demo that converts private documents into a searchable knowledge base and answers questions through retrieval-augmented generation. The project is built for technical showcase, fast review, and practical explanation of the full RAG pipeline.

## Project Positioning

This repository is not just a UI wrapper around an LLM API. It demonstrates a complete document QA workflow:

- ingest local PDF documents
- split long text into retrieval-friendly chunks
- generate embeddings with an open-source model
- index and search content with FAISS
- optionally rerank retrieved results with a cross-encoder
- assemble grounded prompts for answer generation
- connect to an external OpenAI-compatible LLM endpoint

The codebase is intentionally compact so reviewers can understand the architecture quickly.

## Core Highlights

- **End-to-end RAG pipeline**: covers ingestion, chunking, embedding, indexing, retrieval, reranking, prompt construction, and answer generation
- **Local knowledge base workflow**: suitable for private PDFs and offline document preparation
- **Two-stage retrieval option**: combines FAISS recall with cross-encoder reranking to improve answer relevance
- **Clean public repository design**: excludes local indexes, PDFs, notes, and private environment data
- **Demo-friendly interface**: Streamlit UI makes the system easy to present during review or interviews
- **Practical extensibility**: can be extended to multi-document search, uploads, citations, and evaluation

## Technical Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- Sentence Transformers CrossEncoder
- PyPDF
- Requests

## System Workflow

```markdown
Local PDF files
  -> PDF loading
  -> text chunking
  -> embedding generation
  -> FAISS vector indexing
  -> similarity retrieval
  -> optional reranking
  -> grounded prompt assembly
  -> external LLM response
```

## Repository Structure

```markdown
rag-pdf-qa-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── pdfs/                # local PDF input directory
└── faiss_index/         # generated after building the vector index
```

## Why This Project Works Well for Review

- It shows an actual RAG architecture instead of a single LLM call.
- It reflects understanding of retrieval quality, not only generation.
- It uses mainstream open-source tooling seen in real prototype stacks.
- It is small enough to inspect quickly but complete enough to discuss depth.
- It avoids leaking private data by excluding local documents and generated artifacts.

## Functional Capabilities

### 1. Local PDF ingestion
The system reads PDF files from the local `pdfs/` directory and prepares them for downstream retrieval.

### 2. Text chunking
Long documents are segmented into smaller chunks to improve embedding quality and retrieval precision.

### 3. Embedding generation
Text chunks are converted into vectors using a Hugging Face embedding model.

### 4. Vector search with FAISS
The system stores document vectors in a local FAISS index and retrieves relevant chunks for each question.

### 5. Optional reranking
An additional cross-encoder reranker can be enabled to reorder retrieved candidates and improve final context quality.

### 6. Prompt grounding
The app assembles retrieved context into a structured prompt so the model answers from evidence instead of free-form guessing.

### 7. External LLM integration
The final answer is generated through an OpenAI-compatible API, configured via environment variables or the UI.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare documents

Place one or more PDF files in the `pdfs/` directory.

### 3. Configure optional LLM access

If you want generated final answers, provide these environment variables:

```bash
RAG_API_BASE=https://your-api-base.example/v1
RAG_API_KEY=your_api_key
RAG_MODEL_NAME=your_model_name
```

If no API settings are provided, the project can still demonstrate retrieval and prompt assembly.

### 4. Run the application

```bash
streamlit run app.py
```

## Typical Demo Flow

1. Launch the Streamlit application.
2. Select a local PDF from the sidebar.
3. Build or rebuild the vector index.
4. Ask questions in natural language.
5. Inspect the retrieved reference chunks below the answer area.
6. Compare basic retrieval and reranked retrieval behavior.

## Configurable Parameters

The app exposes retrieval-related parameters directly in the sidebar:

- `Chunk Size`
- `Chunk Overlap`
- `Top-K`
- embedding model name
- reranker enable/disable
- reranker model name
- candidate recall size (`Fetch-K`)

These controls make the project useful for demonstrations, experiments, and basic retrieval tuning.

## Example Questions

- What is the main topic of this PDF?
- Summarize the core functions described in the document.
- What installation or configuration process is mentioned?
- Which parameters, hardware models, or specifications appear in the file?

## Privacy and Repository Hygiene

This public version is intentionally cleaned for sharing:

- local PDF files are not committed
- generated FAISS indexes are not committed
- virtual environments are not committed
- text notes and private working files are not committed
- API keys are not stored in the repository
- local absolute paths have been removed from the public code path

## Future Improvements

- multi-document indexing and filtering
- drag-and-drop PDF upload
- highlighted source citations in answers
- separate indexes for different document groups
- retrieval evaluation and benchmarking
- conversation memory across sessions

## Use Cases

- RAG learning and experimentation
- portfolio and resume project showcase
- interview walkthrough for document QA systems
- lightweight internal knowledge retrieval prototype

## License

This repository is provided for learning, demonstration, and portfolio use.
