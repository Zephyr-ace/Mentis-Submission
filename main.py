#!/usr/bin/env python3

from core.chat import Chat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Check if required environment variables are set
    if not os.getenv("USER_ID"):
        print("L USER_ID environment variable is required")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("L OPENAI_API_KEY environment variable is required")
        return
    
    print("> Chat system initialized with 3 retrievers!")
    print("Type 'quit' to exit\n")
    
    # Initialize chat system
    with Chat() as chat:
        while True:
            try:
                # Get user input
                user_query = input("=Ask me anything about your diary: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("=K Goodbye!")
                    break
                
                if not user_query:
                    print("Please enter a question.")
                    continue
                
                print("\n= Searching across all retrievers...")
                
                # Get responses from all 3 retrievers
                responses = chat.chat(user_query)
                
                # Display results
                print("\n" + "="*80)
                print("=� RESULTS FROM 3 DIFFERENT RETRIEVERS")
                print("="*80)
                
                print(f"\n<� MAIN RETRIEVER (Semantic Search):")
                print("-" * 50)
                print(responses["main_retriever"])
                
                print(f"\n=� SIMPLE RAG (Text Chunks):")
                print("-" * 50)
                print(responses["simple_rag"])
                
                print(f"\n=� SUMMARY RAG (Summarized Content):")
                print("-" * 50)
                print(responses["summary_rag"])
                
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n=K Goodbye!")
                break
            except Exception as e:
                print(f"L Error: {e}")
                print("Please try again.\n")

if __name__ == "__main__":
    main()