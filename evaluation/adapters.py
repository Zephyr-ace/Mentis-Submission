"""
Ragas adapters for Mentis retrieval systems.

Wraps existing SimpleRag and SummaryRag retrievers to implement 
the Ragas retriever interface for evaluation.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod

from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.retriever import Retriever


class MentisRetrieverAdapter:
    """
    Adapter that wraps Mentis retrievers to implement Ragas retriever interface.
    
    Ragas expects retrievers to have a retrieve(query, top_k) method that returns
    a list of retrieved documents/chunks.
    """
    
    def __init__(self, retriever_name: str):
        """
        Initialize adapter with specified retriever.
        
        Args:
            retriever_name: Either 'simple_rag', 'summary_rag', or 'main_retriever'
        """
        self.retriever_name = retriever_name
        
        if retriever_name == 'simple_rag':
            self._retriever = SimpleRag()
        elif retriever_name == 'summary_rag':
            self._retriever = SummaryRag()
        elif retriever_name == 'main_retriever':
            self._retriever = Retriever()
        else:
            raise ValueError(f"Unknown retriever: {retriever_name}. Must be 'simple_rag', 'summary_rag', or 'main_retriever'")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k documents for a query.
        
        This method implements the Ragas retriever interface.
        
        Args:
            query: Search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document strings
        """
        try:
            if self.retriever_name == 'main_retriever':
                # Main retriever returns dict, we need to extract text content
                result_dict = self._retriever.retrieve(query)
                results = []
                for category, items in result_dict.items():
                    if isinstance(items, list):
                        for item in items[:top_k//len(result_dict)]:
                            if hasattr(item, 'original_text'):
                                results.append(item.original_text)
                            elif isinstance(item, str):
                                results.append(item)
                return results[:top_k]
            else:
                # SimpleRag and SummaryRag have the same interface: retrieve(query, limit)
                results = self._retriever.retrieve(query, limit=top_k)
                return results
        except Exception as e:
            print(f"Error in {self.retriever_name} retrieval: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry - delegate to underlying retriever"""
        if hasattr(self._retriever, '__enter__'):
            self._retriever.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - delegate to underlying retriever"""
        if hasattr(self._retriever, '__exit__'):
            self._retriever.__exit__(exc_type, exc_val, exc_tb)


class SimpleRagAdapter(MentisRetrieverAdapter):
    """Convenience adapter specifically for SimpleRag"""
    
    def __init__(self):
        super().__init__('simple_rag')


class SummaryRagAdapter(MentisRetrieverAdapter):
    """Convenience adapter specifically for SummaryRag""" 
    
    def __init__(self):
        super().__init__('summary_rag')


class MainRetrieverAdapter(MentisRetrieverAdapter):
    """Convenience adapter specifically for main Retriever"""
    
    def __init__(self):
        super().__init__('main_retriever')


def get_retriever_adapter(retriever_name: str) -> MentisRetrieverAdapter:
    """
    Factory function to create retriever adapters.
    
    Args:
        retriever_name: Either 'simple_rag' or 'summary_rag'
        
    Returns:
        Configured MentisRetrieverAdapter instance
    """
    return MentisRetrieverAdapter(retriever_name)


def get_all_adapters() -> Dict[str, MentisRetrieverAdapter]:
    """
    Get all available retriever adapters.
    
    Returns:
        Dictionary mapping retriever names to adapter instances
    """
    return {
        'simple_rag': SimpleRagAdapter(),
        'summary_rag': SummaryRagAdapter(),
        'main_retriever': MainRetrieverAdapter()
    }


if __name__ == "__main__":
    # Test the adapters
    test_query = "What did the user do yesterday?"
    
    for name, adapter in get_all_adapters().items():
        try:
            with adapter:
                results = adapter.retrieve(test_query, top_k=3)
                print(f"Retrieved {len(results)} documents")
                if results:
                    print(f"First result preview: {results[0][:100]}...")
        except Exception as e:
            print(f"Error testing {name}: {e}")