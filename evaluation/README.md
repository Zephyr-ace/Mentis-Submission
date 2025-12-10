# Retrieval Evaluation System

This directory contains the ragas-based evaluation system for Mentis retrieval components.

## Files

- `adapters.py` - Ragas-compatible adapters for SimpleRag and SummaryRag
- `queries.json` - Evaluation queries for testing retrievers  
- `results_*.json` - Evaluation results for each retriever

## Usage

Simply run the evaluation:

```bash
make eval
```

This will evaluate both retrieval systems using LLM-based context relevance scoring.

## Metrics

The evaluation reports:
- **Context Relevance**: How well retrieved contexts answer the queries (0.0-1.0)
  - Uses an LLM to judge relevance
  - No manual gold standards required
  - Scores: 0.9+ excellent, 0.7-0.9 good, 0.5-0.7 moderate, <0.5 poor

## Results Format

Results are saved as JSON files with:
- `retriever`: Name of the retrieval system
- `num_queries`: Number of queries evaluated
- `nv_context_relevance`: Context relevance score (0.0-1.0)
- `raw_results`: Detailed ragas output

## Integration

Import evaluation results programmatically:

```python
from tests.test_retrieval_eval import get_metrics
results = get_metrics()
```