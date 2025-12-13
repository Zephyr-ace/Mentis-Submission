from core.embedder import TextEmbedder
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

embedder = TextEmbedder()
def encode():

    with open('words.txt', 'r') as file:
        for line in file:
            word = line.strip()
            vector = embedder.embed_text(word) # embedd a single word
            np.save(f"""embeddings/{word}.npy""", vector) # Store a single embedding










def vector_search(query_vector: np.ndarray, k) -> float:
    close_vectors = {}
    for filename in os.listdir("embeddings/"):
        embedding = np.load(f"""embeddings/{filename}""")
        embedding = embedding.reshape(1, -1)
        query_vector_reshaped = query_vector.reshape(1, -1)
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector_reshaped, embedding)[0][0]
        distance = np.linalg.norm(query_vector_reshaped - embedding)

        if similarity > 0.8:
            close_vectors[filename] = (similarity, distance)

    sorted_results = sorted(close_vectors.items(), key=lambda item: item[1], reverse=True)
    top_k_results = sorted_results[:k]

    return close_vectors, top_k_results



def compare_vectors(v1, v2): # compare semantic similarity of two words


    similarity = cosine_similarity(v1, v2)[0][0]
    distance = np.linalg.norm(v1 - v2)

    print(similarity)
    print(distance)

    return cosine_similarity, distance


w1 = np.array(embedder.embed_text("color")).reshape(1, -1)
w2 = np.array(embedder.embed_text("love")).reshape(1, -1)
w3 = np.array(embedder.embed_text("math")).reshape(1, -1)
w4 = np.array(embedder.embed_text("")).reshape(1, -1)




compare_vectors(w1,w2)
compare_vectors(w3,w4)



