import warnings

# Suppress all Python warnings
warnings.filterwarnings("ignore")

from agentic_rag import Agent

Agent = Agent()
user_prompt = input("Enter your message: ")
answer = Agent._answer(user_prompt)
print("\nAnswer:", answer)
