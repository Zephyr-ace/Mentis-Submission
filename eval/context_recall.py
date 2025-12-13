import json
from datasets import Dataset
from ragas.metrics import context_recall
from ragas import evaluate

# Load data from the JSON file
with open('context_recall_data_samples.json', 'r') as file:
    data_samples = json.load(file)

# Create the Dataset from the loaded data
dataset = Dataset.from_dict(data_samples)

# Evaluate the dataset with context recall
score = evaluate(dataset, metrics=[context_recall])

# Convert results to pandas DataFrame and print
results = score.to_pandas()
print(results)
