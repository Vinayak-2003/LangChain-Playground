from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# Intsy Backend - Enterprise API

Enterprise-grade Python FastAPI backend for the Intsy AI Voice Interview Platform.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Poetry & Modern Python Setup](#poetry--modern-python-setup)
4. [Authentication](#authentication)
5. [Role-Based Access Control](#role-based-access-control-rbac)
6. [API Endpoints](#api-endpoints)
7. [Development](#development)
8. [Configuration](#configuration)
9. [Monitoring](#monitoring)

## Quick Start

### Option A: Using Poetry (Recommended)

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the application
python run.py
```

### Option B: Using pip (Legacy)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print("len of chunks:", len(chunks))
print(chunks)
