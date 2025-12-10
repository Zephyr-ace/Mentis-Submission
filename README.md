# Mentis

Personal knowledge retrieval system with RAG evaluation capabilities.

## Quick Start

```bash
# Install dependencies
make install

# Run evaluation
make eval
```

## Architecture

- **SimpleRag**: Vector similarity retrieval
- **SummaryRag**: Summary-based retrieval
- **Evaluation**: LLM-based quality assessment using ragas

## Requirements

- Python 3.12+
- OpenAI API access
- Weaviate