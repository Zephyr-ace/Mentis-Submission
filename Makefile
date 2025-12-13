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



# Encode data for all RAG-systems
encode-all:
	python -c "from core.encoder import Encoder; from rag.simple_rag import SimpleRag; from rag.summaryRag import SummaryRag; SimpleRag().encode(); SummaryRag().encode(); Encoder().encode(store = True, diary_text = open('data/diary.txt').read())"

# Encode data using main encoder
encode:
	python -c "from core.encoder import Encoder; Encoder().encode(store = True, diary_text = open('data/diary.txt').read())"

# Encode data for simple RAG
encode-simple:
	python -c "from rag.simple_rag import SimpleRag; SimpleRag().encode()"

# Encode data for summary RAG
encode-summary:
	python -c "from rag.summaryRag import SummaryRag; SummaryRag().encode()"

# Start chat interface
chat:
	python main.py

