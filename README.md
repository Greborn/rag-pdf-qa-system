# RAG PDF QA System

A lightweight local PDF question-answering system built around the RAG (Retrieval-Augmented Generation) workflow. The project demonstrates how to turn unstructured PDF documents into a searchable knowledge base, retrieve relevant context with vector search, optionally rerank results, and generate final answers through an external LLM API.

## Overview

This project is designed as a concise but complete RAG demo suitable for learning, portfolio presentation, and technical review. It focuses on the full retrieval pipeline rather than product-level complexity.

Core workflow:

1. Load local PDF files.
2. Split long text into smaller chunks.
3. Convert chunks into embeddings with a Hugging Face model.
4. Store and retrieve vectors with FAISS.
5. Optionally rerank retrieved chunks with a cross-encoder.
6. Assemble a grounded prompt from retrieved context.
7. Call an OpenAI-compatible external LLM API to generate the final answer.

## Features

- Local PDF ingestion with `PyPDFLoader`
- Configurable chunking strategy
- Vector retrieval powered by FAISS
- Optional two-stage retrieval with reranking
- Streamlit chat-style interface
- External LLM integration through OpenAI-compatible API endpoints
- Environment-variable based API configuration
- Simple structure for fast understanding and demonstration

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Embeddings
- Sentence Transformers CrossEncoder
- PyPDF
- Requests

## Retrieval Architecture

```markdown
PDF files
  -> document loading
  -> text chunking
  -> embedding generation
  -> FAISS vector index
  -> similarity retrieval
  -> optional cross-encoder reranking
  -> prompt assembly
  -> external LLM answer generation
```

## Project Structure

```markdown
rag-pdf-qa-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── pdfs/                # put your local PDF files here
└── faiss_index/         # generated locally after building the vector index
```

## Why This Project Is Strong for Review

- Demonstrates an end-to-end RAG pipeline rather than isolated API calls
- Separates retrieval and generation clearly, which makes the architecture easy to explain
- Uses practical open-source components commonly seen in real prototyping workflows
- Includes optional reranking, which shows awareness of retrieval quality optimization
- Keeps the implementation compact enough for reviewers to inspect quickly

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare documents

Put one or more PDF files into the `pdfs/` directory.

### 3. Configure optional API access

If you want the app to generate final answers through an external LLM, set these environment variables:

```bash
RAG_API_BASE=https://your-api-base.example/v1
RAG_API_KEY=your_api_key
RAG_MODEL_NAME=your_model_name
```

If these values are not configured, the system can still demonstrate retrieval and prompt assembly.

### 4. Run the app

```bash
streamlit run app.py
```

## Usage

1. Start the Streamlit app.
2. Select a PDF from the sidebar.
3. Build or rebuild the vector index.
4. Ask questions in the chat input.
5. Inspect retrieved context blocks below the chat area.

## Configurable Parameters

The sidebar supports adjusting the main retrieval parameters:

- `Chunk Size`
- `Chunk Overlap`
- `Top-K`
- Embedding model name
- Whether reranking is enabled
- Reranker model name
- Fetch-K for candidate recall

These controls make it easy to compare retrieval strategies during demos.

## Typical Demo Questions

- What is the main topic of this PDF?
- Summarize the core functions described in the document.
- What installation or configuration steps are mentioned?
- What important parameters, models, or hardware details appear in the document?

## Notes

- `faiss_index/` is generated locally and should not be committed.
- `pdfs/` is intentionally excluded from version control to avoid uploading private documents.
- API keys are never stored in the repository.
- This repository is intended as a clean public demo version.

## Future Improvements

- Support multi-document indexing and source filtering
- Add file upload support in the UI
- Add source citation highlighting in answers
- Persist separate indexes for different document sets
- Introduce evaluation metrics for retrieval quality

## License

This project is provided for learning, demonstration, and portfolio use.
