import warnings
warnings.filterwarnings("ignore") # aesthetics

from core.retriever import Retriever
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.llm import LLM_OA
from config.prompts import finalGenerationPrompt, promptRagDecision

def graph_rag_retrieve(question):
    with Retriever() as retriever:
        retrieval_output = retriever.retrieve(question)
        context = retriever.graph_format_to_text(retrieval_output)
        print(context)

# Example usage
graph_rag_retrieve("who is anne")
