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

## Helps

### Install LibreOffice for doc/docx files

```bash
brew install --cask libreoffice # macOS
sudo apt-get install libreoffice # Ubuntu
```

### Install basic dependencies for each type of file

```bash
uv add unstructured pdfplumber python-docx python-pptx markdown openpyxl pandas
```

### Install SpacyTextSplitter

[spaCy](https://spacy.io/usage)

```bash
uv add setuptools wheel
uv add 'spacy[apple]' # for Apple Silicon
python -m spacy download zh_core_web_sm # Chinese
python -m spacy download en_core_web_sm # English
```

### Install chromadb

```bash
uv add chromadb
```

### Install rank_bm25 and jieba

```bash
uv add rank_bm25
uv add jieba # for Chinese
```
