#!/usr/bin/env python3
"""
Targeted evaluation with Anne Frank diary-specific queries designed to show each RAG system's strengths
"""
import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.legacy_eval import Evaluation
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.retriever import Retriever

def evaluate_single_query(rag_system, query, system_name):
    """Evaluate a single query and return detailed scores"""
    print(f"\nEvaluating {system_name} with query: '{query[:60]}...'")
    
    try:
        # Get retrieval results - handle different return formats
        with rag_system:
            if system_name == "MainRetriever":
                result_dict = rag_system.retrieve(query)
                # Extract text content from the complex result structure
                results = []
                if isinstance(result_dict, dict) and "results" in result_dict:
                    for category, items in result_dict["results"].items():
                        for item_tuple in items[:2]:  # Limit per category
                            if isinstance(item_tuple, tuple) and len(item_tuple) >= 2:
                                item = item_tuple[0]
                                if hasattr(item, 'description'):
                                    results.append(f"{category}: {item.description}")
                                elif hasattr(item, 'content'):
                                    results.append(f"{category}: {item.content}")
                                elif hasattr(item, 'original_text'):
                                    results.append(f"{category}: {item.original_text}")
                                elif isinstance(item, str):
                                    results.append(f"{category}: {item}")
                # Limit to 3 total results
                results = results[:3]
            else:
                results = rag_system.retrieve(query, limit=3)
        
        print(f"Retrieved {len(results)} results")
        if results and len(results) > 0:
            print(f"First result preview: {str(results[0])[:100]}...")
        
        if not results:
            return {"relevance": [], "utility": [], "combined": 0.0, "avg_relevance": 0.0, "avg_utility": 0.0}
        
        # Evaluate each result
        evaluator = Evaluation(query)
        eval_results = evaluator._eval_list(results)
        
        relevance_scores = [score[0] for score in eval_results]
        utility_scores = [score[1] for score in eval_results]
        combined_scores = [(r + u) / 4.0 for r, u in eval_results]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_utility = sum(utility_scores) / len(utility_scores) 
        avg_combined = sum(combined_scores) / len(combined_scores)
        
        print(f"  Relevance scores: {relevance_scores} (avg: {avg_relevance:.2f})")
        print(f"  Utility scores: {utility_scores} (avg: {avg_utility:.2f})")
        print(f"  Combined score: {avg_combined:.3f}")
        
        return {
            "relevance": relevance_scores,
            "utility": utility_scores,
            "combined": avg_combined,
            "avg_relevance": avg_relevance,
            "avg_utility": avg_utility
        }
    except Exception as e:
        print(f"  Error evaluating {system_name}: {e}")
        return {"relevance": [], "utility": [], "combined": 0.0, "avg_relevance": 0.0, "avg_utility": 0.0}

def main():
    # Carefully crafted queries for Anne Frank's diary that should favor different systems:
    
    # Queries that should favor MainRetriever (entity-specific, structured):
    entity_queries = [
        "Tell me about Anne's relationships with specific people like Lies, Sanne, and Jopie",
        "What events happened during Anne's birthday celebration?",
        "What emotions did Anne express about school and teachers?", 
        "What problems was Anne dealing with regarding Jewish restrictions?"
    ]
    
    # Queries that should favor SummaryRag (thematic, semantic):
    thematic_queries = [
        "How did Anne feel about turning 13 and growing up?",
        "What was Anne's overall experience with school and education?",
        "How did the war and Nazi restrictions affect Anne's daily life?",
        "What were Anne's thoughts about friendship and social relationships?"
    ]
    
    # Queries that should favor SimpleRag (direct text search):
    direct_queries = [
        "What gifts did Anne receive for her birthday?",
        "Who is Moortje and what role does Moortje play?",
        "What books did Anne mention reading or wanting to read?",
        "What activities did Anne do at school during recess?"
    ]
    
    all_queries = {
        "Entity-focused (MainRetriever advantage)": entity_queries,
        "Thematic (SummaryRag advantage)": thematic_queries, 
        "Direct text (SimpleRag advantage)": direct_queries
    }
    
    # Define all RAG systems
    rag_systems = {
        "SimpleRag": SimpleRag(),
        "SummaryRag": SummaryRag(),
        "MainRetriever": Retriever()
    }

    # Results storage
    final_results = {}
    
    for query_type, queries in all_queries.items():
        print(f"\n{'='*80}")
        print(f"TESTING {query_type.upper()}")
        print(f"{'='*80}")
        
        type_results = {}
        
        for system_name, rag_system in rag_systems.items():
            print(f"\n{'-'*50}")
            print(f"SYSTEM: {system_name}")
            print(f"{'-'*50}")
            
            system_scores = []
            system_relevance = []
            system_utility = []
            
            for query in queries:
                result = evaluate_single_query(rag_system, query, system_name)
                system_scores.append(result["combined"])
                system_relevance.append(result["avg_relevance"])
                system_utility.append(result["avg_utility"])
            
            # Calculate averages for this query type
            avg_combined = sum(system_scores) / len(system_scores) if system_scores else 0
            avg_relevance = sum(system_relevance) / len(system_relevance) if system_relevance else 0
            avg_utility = sum(system_utility) / len(system_utility) if system_utility else 0
            
            type_results[system_name] = {
                "combined": avg_combined,
                "relevance": avg_relevance,
                "utility": avg_utility
            }
            
            print(f"\n{system_name} SUMMARY for {query_type}:")
            print(f"  Relevance: {avg_relevance:.3f}")
            print(f"  Utility: {avg_utility:.3f}")
            print(f"  Combined: {avg_combined:.3f}")
        
        final_results[query_type] = type_results
        
        # Show winner for this category
        winner = max(type_results.keys(), key=lambda x: type_results[x]["combined"])
        print(f"\n🏆 WINNER for {query_type}: {winner} (Score: {type_results[winner]['combined']:.3f})")

    print(f"\n{'='*80}")
    print("FINAL ANALYSIS - EACH SYSTEM'S STRENGTHS")
    print(f"{'='*80}")
    
    for query_type, type_results in final_results.items():
        print(f"\n{query_type}:")
        sorted_systems = sorted(type_results.items(), key=lambda x: x[1]["combined"], reverse=True)
        for i, (system, scores) in enumerate(sorted_systems):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {rank} {system:15} | Combined: {scores['combined']:.3f} | Relevance: {scores['relevance']:.3f}")

if __name__ == "__main__":
    main()