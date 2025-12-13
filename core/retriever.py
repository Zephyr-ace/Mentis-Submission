from core.llm import LLM_OA
from config.classes import QueriesAndClassification, FilteredResults
from config.prompts import promptQueryRewriteAndClassify, promptFilterResults
from core.vector_db import VectorDB
from pydantic import BaseModel
import os
from core.schema_generator import discover_collections_in_module
import config.classes as classes
import weaviate.classes as wvc


class Retriever:
    def __init__(self):
        self.llm = LLM_OA("gpt-5-mini")
        self.user_id = os.getenv("USER_ID")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    
    def retrieve(self, user_prompt: str, min_score: float = 0.3) -> dict:
        """Main retrieve method that orchestrates the full pipeline"""
        
        # Step 1: Generate rewritten queries with classifications in one call
        result = self.llm.generate_structured(
            prompt=promptQueryRewriteAndClassify.replace("<USER_MESSAGE>", user_prompt),
            desired_output_format=QueriesAndClassification
        )
        
        query_pairs = [(item.rewritten_query, item.query_category) for item in result.items]
        with VectorDB(user_id=self.user_id) as db:

            # Step 2: Search in parallel with score threshold
            categoric_search_results = db.parallel_hybrid_search(query_pairs, min_score=min_score)

            # Step 3 filter results (disabled for now; use all search results directly)
            filtered_categoric_results = {
                category: results.copy() for category, results in categoric_search_results.items()
            }

            # Now use the filtered results to get connected objects
            all_object_ids = []
            for category, results in filtered_categoric_results.items():
                for model_instance, score in results:
                    all_object_ids.append(model_instance.object_id)
            
            print(f"Found {len(all_object_ids)} objects to check for connections: {all_object_ids[:3]}...")
            
            # Step 5: Fetch connected objects
            connected_objects = db.get_connected_objects(all_object_ids)
            print(f"Found connections for {len(connected_objects)} objects")
        
        # Step 6: Merge connected objects into filtered categoric results
        enhanced_results = self._merge_connected_objects(filtered_categoric_results, connected_objects)

        return enhanced_results

    def _merge_connected_objects(self, original_results: dict[str, list], connected_objects: list) -> dict[str, list]:
        """
        Merge connected objects into the original search results.
        Connected objects are added to appropriate categories based on their type.
        """
        # Create a copy to avoid modifying the original results
        enhanced_results = {}
        for category, results in original_results.items():
            enhanced_results[category] = results.copy()
        
        # Mapping from model types to category names
        type_to_category = {
            'Event': 'Event',
            'Person': 'Person', 
            'ThoughtReflection': 'ThoughtReflection',
            'Emotion': 'Emotion',
            'Problem': 'Problem',
            'Achievement': 'Achievement',
            'FutureIntention': 'FutureIntention'
        }
        
        # Keep track of already included object IDs to avoid duplicates
        existing_object_ids = set()
        for category, results in enhanced_results.items():
            for model_instance, score in results:
                existing_object_ids.add(model_instance.object_id)
        
        # Add connected objects to their appropriate categories
        for connected_obj in connected_objects:
            # Skip if this object is already in results
            if connected_obj.object_id in existing_object_ids:
                continue
            
            # Determine which category this connected object belongs to
            model_type = connected_obj.__class__.__name__
            target_category = type_to_category.get(model_type)
            
            if target_category:
                # Initialize category if it doesn't exist
                if target_category not in enhanced_results:
                    enhanced_results[target_category] = []
                
                # Add connected object with a default score (lower than search results)
                # This indicates it's a connected object, not a direct search result
                enhanced_results[target_category].append((connected_obj, 0.1))
                existing_object_ids.add(connected_obj.object_id)
        
        return enhanced_results

    def graph_format_to_text(self, retrieval_output) -> str:
        """
        Convert graph retriever output into plain text for downstream models.
        Accepts either the full retrieval response (with 'results') or just the
        results dictionary keyed by category.
        """
        if not retrieval_output:
            return "No relevant information found."

        # The retrieval pipeline returns a dict keyed by category; if a wrapped
        # response with a "results" key ever arrives, unwrap it, otherwise use
        # the dict directly.
        if isinstance(retrieval_output, dict):
            main_results = retrieval_output.get("results") or retrieval_output
        else:
            main_results = retrieval_output
        if not main_results:
            return "No relevant information found."

        formatted_results = []
        for category, items in main_results.items():
            if not items:
                continue

            for item_with_score in items:
                model_instance = item_with_score[0] if isinstance(item_with_score, tuple) else item_with_score

                if hasattr(model_instance, 'title') and hasattr(model_instance, 'description'):
                    formatted_results.append(f"{category}: {model_instance.title} - {model_instance.description}")
                elif hasattr(model_instance, 'name') and hasattr(model_instance, 'description'):
                    formatted_results.append(f"{category}: {model_instance.name} - {model_instance.description}")
                elif hasattr(model_instance, 'content'):
                    formatted_results.append(f"{category}: {model_instance.content}")
                else:
                    formatted_results.append(f"{category}: {str(model_instance)}")

        return "\n\n".join(formatted_results) if formatted_results else "No relevant information found."
