from config.classes import ChunkSummary
from core.vector_db import VectorDB #  • VectorDB: Handles vector storage, embedding generation, and semantic search
import os # for environment variables
from utils.functions import *
from core.llm import LLM_OA
from config.prompts import promptSummarize
import json
from pathlib import Path



class SummaryRag():
    def __init__(self):
        self.user_id = os.getenv("USER_ID")
        if not self.user_id:
            raise EnvironmentError("SimpleRag requires USER_ID environment variable to be set")
        self.promptSummarize = promptSummarize
        self.llm_oa = LLM_OA("gpt-4.1-mini-2025-04-14") # pinned: consistent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


    def _diary_to_chunks(self, diary: str) -> list[str]:
        """
        Split diary by date lines (e.g., "Sunday , 14 June , 1942")
        Each chunk starts with its date line and contains the following entry content
        """
        import re
        
        # Pattern to match date lines: Weekday , Day Month , Year
        date_pattern = r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*,.*\d{4}$'
        
        lines = diary.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            if re.match(date_pattern, line.strip()):
                # If we have accumulated content, save it as a chunk
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if len(chunk_text) > 10:  # Only keep substantial chunks
                        chunks.append(chunk_text)
                # Start new chunk with this date line
                current_chunk = [line]
            else:
                # Add line to current chunk
                if current_chunk:  # Only add if we're in a chunk
                    current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) > 10:
                chunks.append(chunk_text)
        
        return chunks


    def _summarizer(self, raw_chunks: list[str], batch_size: int|None)-> list[ChunkSummary]:
        """each chunk will get summarized in parallel/batches"""
        print("start summarizing chunks...")
        prompts: list[str] = [self.promptSummarize + "\n\n" + chunk for chunk in raw_chunks]
        if batch_size is not None:
            summarized_chunks = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                summarized_chunks += self.llm_oa.generate_structured_parallel_sync(batch, ChunkSummary)
        else:
            summarized_chunks = self.llm_oa.generate_structured_parallel_sync(prompts, ChunkSummary)
        print("end summarizing chunks! \n\n")
        return summarized_chunks



    # Helpers
    def _cache_exists(self, cache_file: str) -> bool:
        return os.path.exists(cache_file)

    def _load_temp_cache(self, cache_file: str) -> list[ChunkSummary]:
        print("Loading processed chunks from cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChunkSummary(**e) for e in data]

    def _save_temp_cache(self, chunks: list[ChunkSummary], cache_file: str) -> None:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the "data" folder exists
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4, default=str, ensure_ascii=False)

    def _store_in_weaviate(self, chunks: list[ChunkSummary], replace: bool = True) -> None:
        try:
            with VectorDB() as db:
                collection_name = "Summary"
                
                # Delete only the Summary collection if replace = True
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



    def encode(self, diary: str = None, batch_size: int|None = 10, cache_file: str = "data/summarized_diary.json") -> list[ChunkSummary]:
        """
        forward function. combines all local methods into global
        (ps: combines both encoder and retrieval functions of Rag)
        """
        # If no diary provided, read from default file
        if diary is None:
            with open('data/diary.txt', 'r') as f:
                diary = f.read()
        
        # Always use existing cache if it exists - never replace cache
        if self._cache_exists(cache_file):
            chunks = self._load_temp_cache(cache_file)
            print("storing in weaviate")
            self._store_in_weaviate(chunks, replace=True)  # Always replace collection
            return chunks
        
        # Only create new cache if it doesn't exist
        raw_chunks = self._diary_to_chunks(diary)
        chunks = self._summarizer(raw_chunks, batch_size=batch_size)
        
        # Save to cache file (new cache creation)
        self._save_temp_cache(chunks, cache_file)
        
        print("storing in weaviate")
        self._store_in_weaviate(chunks, replace=True)  # Replace collection only when creating new cache
            
        return chunks

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Find the closest neighbor vectors in the Summary collection for a given query"""
        try:
            with VectorDB() as db:
                results = db.vector_search(
                    collection_name="Summary",
                    query=query,
                    vector_name="content",
                    limit=limit
                )
                return [result.get("content", "") for result in results]
        except Exception as e:
            print(f"❌ Failed to retrieve from Weaviate: {e}")
            return []
