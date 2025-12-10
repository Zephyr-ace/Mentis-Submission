#!/usr/bin/env python3
"""
Legacy evaluation runner for Mentis RAG systems.
"""
import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.legacy_eval import run_evaluation
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from utils.functions import loadText, load_queries

def main():
    # Encoding; set true to encode and store or update the data
    encode = False
    if encode:
        diary_partial = loadText()
        with SimpleRag() as rag:
            rag.encode(diary_partial)
        with SummaryRag() as rag:
            rag.encode(diary_partial)

    # Load queries and run evaluation
    queries = load_queries()

    # Define RAG systems to evaluate
    rag_systems = {
        "SimpleRag": SimpleRag(),
        "SummaryRag": SummaryRag()
    }

    # Run evaluation
    scores = run_evaluation(rag_systems, queries)

    print("Evaluation Results:")
    for system_name, score in scores.items():
        print(f"{system_name}: {score:.3f}")

if __name__ == "__main__":
    main()