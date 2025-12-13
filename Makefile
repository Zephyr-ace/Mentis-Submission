# Makefile for Mentis project

.PHONY: help install eval clean encode-rags encode-main chat

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install project dependencies"
	@echo "  eval        - Run retrieval evaluation"
	@echo "  encode-rags - Encode data for SimpleRag and SummaryRag"
	@echo "  encode 	 - Encode data using main encoder"
	@echo "  chat        - Start chat interface"
	@echo "  clean       - Clean temporary files"

# Install dependencies
install:
	pip install -r requirements.txt

# Run retrieval evaluation
eval: 
	python tests/test_retrieval_eval.py

# Encode data for SimpleRag and SummaryRag
encode-rags:
	python -c "from rag.simple_rag import SimpleRag; from rag.summaryRag import SummaryRag; SimpleRag().encode(); SummaryRag().encode()"

# Encode data using main encoder
encode:
	python -c "from core.encoder import Encoder; Encoder().encode(store = True, diary_text = open('data/diary.txt').read())"

# Start chat interface
chat:
	python main.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f evaluation/results_*.json