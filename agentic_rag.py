from core.retriever import Retriever
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.llm import LLM_OA
from config.prompts import finalGenerationPrompt, promptRagDecision
from config.classes import RAGdecision, TrueFalse


class Agent:
    def __init__(self, model: str = "gpt-5.1"):
        self.retriever = Retriever()
        self.simple_rag = SimpleRag()
        self.summary_rag = SummaryRag()
        self.llm = LLM_OA(model)
        self.llm_simple = LLM_OA("gpt-5-nano")
        self.system_prompt = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Release any network clients."""
        self.llm.close()
        self.llm_simple.close()

    def answer(self, user_prompt):

        rag_necessity = self.llm_simple.generate_structured(
            prompt="Is additional context needed to respond to the users message? users message:" + user_prompt,
            desired_output_format=TrueFalse
        ).answer
        print("RAG necessary? = : ", rag_necessity)

        # Access the specific 'answer' attribute of the TrueFalse model
        if not rag_necessity:
            # Return the answer when using direct generation
            return self.llm.generate(user_prompt)
        else:
            # Retrieve context and generate answer
            context = self._retrieve(user_prompt)

            prompt = finalGenerationPrompt.replace("{user_prompt}", user_prompt).replace("{context}", context)
            answer = self.llm.generate(prompt)
            return answer

    def _retrieve(self, user_prompt: str) -> str:
        """Generate 3 different answers using 3 different retrievers"""
        rag_decision = self.llm.generate_structured(
            prompt = promptRagDecision + user_prompt,
            desired_output_format = RAGdecision
        )
        
        # 1. Retrieve using quickest rag available
        if rag_decision.simpleRAG:
            print("simple Rag")
            results = self.simple_rag.retrieve(user_prompt, limit=5)
            return "\n\n".join(results)

        elif rag_decision.summaryRAG:
            print("summary Rag")
            results = self.summary_rag.retrieve(user_prompt, limit=5)
            return "\n\n".join(results)
        
        elif rag_decision.graphRAG:
            print("graph Rag")
            main_retrieval_output = self.retriever.retrieve(user_prompt)
            main_results = main_retrieval_output.get("results", {}) if isinstance(main_retrieval_output, dict) else main_retrieval_output

            # Convert complex graph results to text format
            formatted_results = []
            for category, items in main_results.items():
                for item, score in items:
                    if hasattr(item, 'title') and hasattr(item, 'description'):
                        formatted_results.append(f"{category}: {item.title} - {item.description}")
                    elif hasattr(item, 'name') and hasattr(item, 'description'):
                        formatted_results.append(f"{category}: {item.name} - {item.description}")
                    elif hasattr(item, 'content'):
                        formatted_results.append(f"{category}: {item.content}")
                    else:
                        formatted_results.append(f"{category}: {str(item)}")

            return "\n\n".join(formatted_results)
        else:
            print("error!!! no RAG system has been picked by the agent")
            return "No relevant information found."
