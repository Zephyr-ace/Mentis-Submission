#!/usr/bin/env python3
"""
Test MainRetriever after proper data storage
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

def test_main_retriever_performance():
    """Test MainRetriever with the same queries after data is properly stored"""
    print("=== TESTING MAINRETRIEVER AFTER PROPER STORAGE ===\n")
    
    # Same queries from the previous evaluation
    test_queries = [
        "What gifts did Anne receive for her birthday?",
        "Tell me about Anne's relationship with her mother", 
        "What emotions did Anne express in her diary?",
        "What events happened during Anne's birthday celebration?",
        "What problems was Anne dealing with regarding Jewish restrictions?"
    ]
    
    try:
        with Retriever() as retriever:
            print("Testing MainRetriever queries...\n")
            
            for i, query in enumerate(test_queries, 1):
                print(f"{i}. Query: '{query}'")
                result = retriever.retrieve(query)
                
                if isinstance(result, dict) and "results" in result:
                    total_results = sum(len(items) for items in result["results"].values())
                    print(f"   📊 Total results: {total_results}")
                    
                    # Show results by category
                    for category, items in result["results"].items():
                        if items:
                            print(f"   {category}: {len(items)} items")
                            # Show best result with more details
                            best_item, best_score = items[0]
                            content = getattr(best_item, 'description', '') or getattr(best_item, 'title', '') or getattr(best_item, 'name', '')
                            print(f"     Best: {content[:80]}... (score: {best_score:.3f})")
                    
                    if "debug_info" in result:
                        debug = result["debug_info"]
                        print(f"   🔍 Debug: {debug['total_objects_searched']} objects searched, {debug['connections_found']} connections found")
                    
                else:
                    print("   ❌ Unexpected result format")
                print("-" * 80)
                
    except Exception as e:
        print(f"❌ Error testing MainRetriever: {e}")
        import traceback
        traceback.print_exc()

def run_comparative_evaluation():
    """Run the same evaluation as before to compare performance"""
    print("\n=== COMPARATIVE EVALUATION WITH PROPER DATA ===\n")
    
    # Targeted queries that should favor different systems
    entity_queries = [
        "Tell me about Anne's relationships with specific people like Lies, Sanne, and Jopie",
        "What events happened during Anne's birthday celebration?",
        "What emotions did Anne express about school and teachers?", 
        "What problems was Anne dealing with regarding Jewish restrictions?"
    ]
    
    print("Testing entity-focused queries (MainRetriever's specialty)...")
    
    systems = {
        "SimpleRag": SimpleRag(),
        "SummaryRag": SummaryRag(), 
        "MainRetriever": Retriever()
    }
    
    results = {}
    
    for system_name, rag_system in systems.items():
        print(f"\n--- {system_name} ---")
        system_scores = []
        
        for query in entity_queries:
            print(f"Query: {query[:50]}...")
            
            try:
                with rag_system:
                    if system_name == "MainRetriever":
                        result_dict = rag_system.retrieve(query)
                        # Extract text content from the complex result structure
                        query_results = []
                        if isinstance(result_dict, dict) and "results" in result_dict:
                            for category, items in result_dict["results"].items():
                                for item_tuple in items[:2]:  # Limit per category
                                    if isinstance(item_tuple, tuple) and len(item_tuple) >= 2:
                                        obj, score = item_tuple[0], item_tuple[1]
                                        if hasattr(obj, 'description'):
                                            query_results.append(f"{category}: {obj.description}")
                                        elif hasattr(obj, 'content'):
                                            query_results.append(f"{category}: {obj.content}")
                        query_results = query_results[:3]
                    else:
                        query_results = rag_system.retrieve(query, limit=3)
                
                if query_results:
                    # Quick evaluation
                    evaluator = Evaluation(query)
                    eval_results = evaluator._eval_list(query_results)
                    avg_relevance = sum(score[0] for score in eval_results) / len(eval_results)
                    avg_combined = sum((score[0] + score[1]) / 4.0 for score in eval_results) / len(eval_results)
                    
                    system_scores.append(avg_combined)
                    print(f"  Relevance: {avg_relevance:.2f}, Combined: {avg_combined:.3f}")
                else:
                    system_scores.append(0.0)
                    print(f"  No results")
                    
            except Exception as e:
                print(f"  Error: {e}")
                system_scores.append(0.0)
        
        avg_score = sum(system_scores) / len(system_scores) if system_scores else 0
        results[system_name] = avg_score
        print(f"{system_name} average score: {avg_score:.3f}")
    
    print(f"\n{'='*60}")
    print("FINAL COMPARISON - ENTITY-FOCUSED QUERIES")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (system, score) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{rank} {system:15} | Score: {score:.3f}")

if __name__ == "__main__":
    test_main_retriever_performance()
    run_comparative_evaluation()