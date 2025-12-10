import os
from typing import List
from core.vector_db import VectorDB
from config.classes import SimpleRagChunk


class SimpleRag:
    def __init__(self):
        self.user_id = os.getenv("USER_ID")
        if not self.user_id:
            raise EnvironmentError("SimpleRag requires USER_ID environment variable to be set")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - overlap
        
        return chunks
    


    def _store_in_weaviate(self, chunks: List[SimpleRagChunk], replace: bool = True) -> None:
        """Store chunks in Weaviate vector database"""
        try:
            with VectorDB() as db:
                collection_name = "simple_rag"
                
                # Delete existing collection if replace = True
                if replace:
                    collections = db.client.collections
                    if collections.exists(collection_name):
                        collections.delete(collection_name)
                        print(f"🗑️ Deleted existing {collection_name} collection")
                        # Recreate the schema and tenant after deletion
                        db._create_schema()
                        db._create_tenant()
                
                db.store_chunks(chunks)
                print(f"✅ Stored {len(chunks)} chunks in Weaviate for user {db.user_id}")
        except Exception as e:
            print(f"❌ Failed to store in Weaviate: {e}")


    # Encoding/storing the data
    def encode(self, diary: str = None, chunk_size: int = 400, overlap: int = 200) -> List[SimpleRagChunk]:
        """
        Encode diary text into chunks and store in vector database
        """
        
        # If no diary provided, read from default file
        if diary is None:
            with open('data/diary.txt', 'r') as f:
                diary = f.read()

        # Split text into overlapping chunks
        chunks = self._chunk_text(diary, chunk_size, overlap)

        if not chunks:
            return []

        # Create SimpleRagChunk instances
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_objects.append(SimpleRagChunk(
                content=chunk_text,
                chunk_index=i
            ))

        # Store in Weaviate
        self._store_in_weaviate(chunk_objects)

        return chunk_objects


    # retrieving encoded/stored data
    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """Find the closest neighbor vectors in the simple_rag collection for a given query"""
        try:
            with VectorDB() as db:
                results = db.vector_search(
                    collection_name="simple_rag",
                    query=query,
                    vector_name="content",
                    limit=limit
                )
                return [result.get("content", "") for result in results]
        except Exception as e:
            print(f"❌ Failed to retrieve from Weaviate: {e}")
            return []