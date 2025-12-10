from core.retriever import Retriever
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.llm import LLM_OA
from config.prompts import finalGenerationPrompt


class Chat:
    def __init__(self):
        self.retriever = Retriever()
        self.simple_rag = SimpleRag()
        self.summary_rag = SummaryRag()
        self.llm = LLM_OA("o3")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def chat(self, user_query: str) -> dict[str, str]:
        """Generate 3 different answers using 3 different retrievers"""
        
        # 1. Retrieve from all 3 sources
        main_retrieval_output = self.retriever.retrieve(user_query)
        main_results = main_retrieval_output.get("results", {}) if isinstance(main_retrieval_output, dict) else main_retrieval_output # extract results with safety fallback
        simple_results = self.simple_rag.retrieve(user_query, limit=5)
        summary_results = self.summary_rag.retrieve(user_query, limit=5)
        
        # 2. Format each result set
        main_context = self._format_main_results(main_results)
        simple_context = self._format_simple_results(simple_results)
        summary_context = self._format_summary_results(summary_results)
        
        # 3. Generate 3 separate answers
        return {
            "main_retriever": self._generate_answer(user_query, main_context),
            "simple_rag": self._generate_answer(user_query, simple_context),
            "summary_rag": self._generate_answer(user_query, summary_context)
        }
    
    def _format_main_results(self, results: dict[str, list]) -> str:
        """Format results from main retriever (structured objects by category)"""
        if not results:
            return "No relevant information found."
        
        formatted_parts = []
        for category, items in results.items():
            if items:
                formatted_parts.append(f"\n--- {category} ---")
                for item in items:
                    # Extract the model instance (item is tuple of (model, score))
                    model_instance = item[0] if isinstance(item, tuple) else item
                    
                    # Format based on model type
                    if hasattr(model_instance, 'title') and hasattr(model_instance, 'description'):
                        formatted_parts.append(f"• {model_instance.title}: {model_instance.description}")
                    elif hasattr(model_instance, 'name') and hasattr(model_instance, 'description'):
                        formatted_parts.append(f"• {model_instance.name}: {model_instance.description}")
                    elif hasattr(model_instance, 'content'):
                        formatted_parts.append(f"• {model_instance.content}")
                    else:
                        formatted_parts.append(f"• {str(model_instance)}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant information found."
    
    def _format_simple_results(self, results: list[str]) -> str:
        """Format results from simple RAG (list of text chunks)"""
        if not results:
            return "No relevant information found."
        
        formatted_parts = ["--- Simple RAG Chunks ---"]
        for i, chunk in enumerate(results, 1):
            formatted_parts.append(f"Chunk {i}: {chunk}")
        
        return "\n".join(formatted_parts)
    
    def _format_summary_results(self, results: list[str]) -> str:
        """Format results from summary RAG (list of summary texts)"""
        if not results:
            return "No relevant information found."
        
        formatted_parts = ["--- Summary Information ---"]
        for i, summary in enumerate(results, 1):
            formatted_parts.append(f"Summary {i}: {summary}")
        
        return "\n".join(formatted_parts)
    
    def _generate_answer(self, user_query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context"""
        prompt = f"""

{finalGenerationPrompt}

User Question: 
<<<{user_query}>>>

The following might be relevant information found in the diary entries:
<<<{context}>>>

now summarize!"""
        
        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"