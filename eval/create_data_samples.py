import warnings

warnings.filterwarnings("ignore") # aesthetics

from core.retriever import Retriever
from rag.simple_rag import SimpleRag
from rag.summaryRag import SummaryRag
from core.llm import LLM_OA
from config.prompts import finalGenerationPrompt, promptRagDecision

import json



# example questions + ground_truths
questions = ["who is my daddy?", "What date did Anne Frank go into hiding?", "Who was Anne Frank's first diary entry addressed to?"]
ground_truths = ["example ground truth1", "example ground truth2"]

def graph_rag_retrieve(question):
    with Retriever() as retriever:
        retrieval_output = retriever.retrieve(question)
        context = retriever.graph_format_to_text(retrieval_output)
        return context

def simple_rag_retrieve(question):
    with SimpleRag() as rag:
        context = rag.retrieve(question)
        return context

def summary_rag_retrieve(question):
    with SummaryRag() as rag:
        context = rag.retrieve(question)
        return context

# Example usage

def answer(question, context):
    context = "\n\n".join(context)
    llm = LLM_OA("gpt-5.1")
    prompt = finalGenerationPrompt.replace("{user_prompt}", question).replace("{context}", context)
    answer = llm.generate(prompt)
    return answer


# write data samples for simple_RAG


data_samples = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

# Collect data samples for each question
for question, ground_truth in zip(questions, ground_truths):
    # Retrieve the context
    context = simple_rag_retrieve(question)  # You can replace this with simple_rag_retrieve or summary_rag_retrieve based on your use case

    # Get the answer
    generated_answer = answer(question, context)

    # Append data to the corresponding lists
    data_samples["question"].append(question)
    data_samples["answer"].append(generated_answer)
    data_samples["contexts"].append([context])  # Assuming context is a list of statements
    data_samples["ground_truth"].append(ground_truth)

# Ensure the directory exists
import os
os.makedirs('data_samples', exist_ok=True)

# Write the data samples to a JSON file with UTF-8 characters preserved
with open('data_samples/simple_rag_data_samples.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_samples, json_file, indent=4, ensure_ascii=False)

print("Data samples have been saved to 'eval/simple_rag_data_samples.json'.")
