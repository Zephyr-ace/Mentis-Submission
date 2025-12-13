import json
from datasets import Dataset
from ragas.metrics import context_recall, context_precision
from ragas import evaluate
import os


# Load data from the JSON file
with open('data_samples/simple_rag_data_sample.json', 'r') as file:
    data_samples = json.load(file)

# Create the Dataset from the loaded data
dataset = Dataset.from_dict(data_samples)

# Evaluate the dataset with context recall and precision
score = evaluate(dataset, metrics=[context_precision, context_recall])


# Convert results to pandas DataFrame and print
results = score.to_pandas()
# Print specific columns: question, context_precision, and context_recall
#print(results[['question', 'context_precision', 'context_recall']])


# Cache the results as json file
os.makedirs('results', exist_ok=True) # Ensure the directory exists
with open('results/evaluation_results.json', 'w') as json_file:
    json.dump(results.to_dict(orient='records'), json_file, indent=4)