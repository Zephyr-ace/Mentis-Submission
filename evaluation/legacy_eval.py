"""
Evaluation module for Mentis retrieval systems.
"""

from core.llm import LLM_OA
from config.prompts import promptEvalPrompt
from config.classes import OutputEval
from typing import List, Dict, Any


class Evaluation:
    """Legacy evaluation class using custom LLM-based scoring."""
    
    def __init__(self, Prompt):
        self.evalPrompt = (promptEvalPrompt)
        self.Prompt = Prompt
        self.llm_oa = LLM_OA("o4-mini")

    def _eval_list(self, retrieved_results: list[str]) -> list[tuple[int, int]]:
        retrieved_results = [self.evalPrompt + "\n\n" + "<question or user prompt> "+self.Prompt + "\n" + "<context>" + "\n" +result for result in retrieved_results]
        eval_results = self.llm_oa.generate_structured_parallel_sync(retrieved_results, OutputEval)
        eval_results = [(result.relevance, result.overallUtility) for result in eval_results]
        return eval_results


def evaluate_rag_system(rag_system: Any, queries: List[str]) -> float:
    """
    Evaluate a RAG system across multiple queries and return average normalized score.
    
    Args:
        rag_system: RAG system instance (must have retrieve method)
        queries: List of query strings to evaluate
    
    Returns:
        Average normalized score between 0 and 1
    """
    all_scores = []
    
    # Create single evaluator instance to reuse LLM
    evaluator = Evaluation("")  # Initialize with empty query
    
    for query in queries:
        evaluator.Prompt = query  # Update the query for this iteration
        results = rag_system.retrieve(query)
        eval_results = evaluator._eval_list(results)
        
        # Calculate combined scores for this query
        for relevance, utility in eval_results:
            combined_score = (relevance + utility) / 4.0  # Normalize to 0-1 range
            all_scores.append(combined_score)
    
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


def run_evaluation(rag_systems: Dict[str, Any], queries: List[str]) -> Dict[str, float]:
    """
    Run evaluation on multiple RAG systems and return normalized scores.
    
    Args:
        rag_systems: Dictionary mapping system names to RAG instances
        queries: List of queries to evaluate
    
    Returns:
        Dictionary mapping system names to their average scores (0-1)
    """
    results = {}
    
    for system_name, rag_system in rag_systems.items():
        with rag_system:
            score = evaluate_rag_system(rag_system, queries)
            results[system_name] = score
    
    return results


