import openai
import os
from typing import List, Union, Dict

class TextEmbedder:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "text-embedding-ada-002"
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """Single text embedding for backward compatibility."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 2048) -> List[List[float]]:
        """Batch embedding for multiple texts. OpenAI supports up to 2048 texts per request."""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            # Extract embeddings in the same order as input
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_text_dict(self, text_dict: Dict[str, str]) -> Dict[str, List[float]]:
        """Embed a dictionary of {key: text} and return {key: embedding}."""
        if not text_dict:
            return {}
        
        # Separate keys and texts while preserving order
        keys = list(text_dict.keys())
        texts = list(text_dict.values())
        
        # Get batch embeddings
        embeddings = self.embed_texts_batch(texts)
        
        # Map back to keys
        return dict(zip(keys, embeddings))