# rag-demo

## Install Requirements

```bash
uv sync
```

## Run the app

```bash
uv run app.py
```

## Dependencies

- `langchain`: the core library for building LLM applications
- `langchain_community`: the community extensions for langchain for RAG
- `pypdf`: for parsing PDF documents
- `sentence-transformers`: for sentence embeddings
- `faiss-cpu`: for vector similarity search
- `dashscope`: for LLM inference
- `openai`: for calling OpenAI/DeepSeek API
