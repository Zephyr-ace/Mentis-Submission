import asyncio
import re
import json
import os
import uuid
from typing import Type, Tuple
from pathlib import Path

from core.llm import LLM_OA
from utils.functions import *            # keep existing helper imports
from config.classes import *              # as before        # Event, Person, …
from core.vector_db import VectorDB
from config.prompts import *              # promptEvents, …
from core.graph import GraphProcessor


class Encoder:
    RETRIES = 2
    PARALLEL_LIMIT = 10 # for rate limits

    def __init__(self):
        self.llm_oa = LLM_OA("gpt-4.1-mini-2025-04-14") # pinned: consistent
        self.prompt_stage_one = (promptStageOne)
        # self.chat_prompt = "..."      # Future: for chat messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _diary_to_chunks(self, diary: str) -> list[str]:
        """
        Split diary by date lines (e.g., "Sunday , 14 June , 1942")
        Each chunk starts with its date line and contains the following entry content
        """
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

    # stage 1
    async def _stage_one(self, raw_chunks: list[str], batch_size: int|None)-> list[DiaryExtraction]:
        """chunk will get processed to determine which information objects to create."""

        prompts: list[str] = [self.prompt_stage_one + "\n\n" + chunk for chunk in raw_chunks]
        if batch_size is not None:
            stage_one_chunks = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                stage_one_chunks += await self.llm_oa.generate_structured_parallel(batch, DiaryExtraction)

        else:
            stage_one_chunks = await self.llm_oa.generate_structured_parallel(prompts, DiaryExtraction)
        return stage_one_chunks


        # stage 2
    async def _stage_two(self, raw_chunks: list[str],stage_one_results: list[DiaryExtraction])-> list[Chunk]:
        """for each chunkExtraction-> information Object"""
        if len(raw_chunks) != len(stage_one_results):
            raise ValueError(
                "raw_chunks and stage_one_results length mismatch "
                f"({len(raw_chunks)} vs {len(stage_one_results)})"
            )
        # Helper Map
        CAT_INFO: dict[str, Tuple[str, Type[BaseModel]]] = {
            "EventAction": (promptEvents, Event),
            "People": (promptPeople, Person),
            "ThoughtReflections": (promptThoughts, ThoughtReflection),
            "Emotion": (promptEmotions, Emotions),
            "Problem": (promptProblems, Problem),
            "Achievement": (promptAchievements, Achievement),
            "FutureIntentions": (promptGoals, FutureIntention),
        }

        def build_prompt(cat_key: str, descriptions: list[str], diary: str) -> str:
            """Assemble the final LLM prompt for one category."""
            cat_prompt, _ = CAT_INFO[cat_key]
            desc_section  = "\n".join(descriptions)
            return (
                f"{cat_prompt}\n\n"
                "Descriptions:\n"
                f"{desc_section}\n\n"
                "Diary entry:\n"
                f"{diary}"
            )

        semaphore = asyncio.Semaphore(self.PARALLEL_LIMIT) # sets parallel limit.

        async def call_llm(prompt: str, out_type: Type[BaseModel],) -> BaseModel | None:
            """Run a single LLM call with retry & back-off."""
            for attempt in range(self.RETRIES):
                try:
                    async with semaphore:
                        return await self.llm_oa.generate_structured_async(
                            prompt, out_type
                        )
                except Exception as exc:
                    if attempt == self.RETRIES - 1:
                        # last attempt – give up
                        print(f"[stage_two] giving up: {type(exc).__name__}: {exc}")
                        return None
                    # back-off: 1s, 2s, 4s, …
                    backoff = 2 ** attempt
                    print(f"[stage_two] retry in {backoff}s – {exc}")
                    await asyncio.sleep(backoff)

            print("every attempt failed! (returning None)")
            return None

        def convert_list(data, ModelCls):
            """Helper function to convert LLM response data to pydantic model list"""
            if data is None:
                return None
            
            # Handle container models like Events, People, etc. that have .items
            if hasattr(data, 'items'):
                items = data.items
                # Ensure each item has proper UUID if it doesn't already have one
                for item in items:
                    if hasattr(item, 'object_id') and (not item.object_id or item.object_id == ""):
                        item.object_id = str(uuid.uuid4())
                return items
            
            # Handle single model instance that should be wrapped in a list
            if isinstance(data, ModelCls):
                # Ensure proper UUID
                if hasattr(data, 'object_id') and (not data.object_id or data.object_id == ""):
                    data.object_id = str(uuid.uuid4())
                return [data]
            
            # Handle list of items that need conversion
            if isinstance(data, list):
                converted_items = []
                for item in data:
                    if isinstance(item, ModelCls):
                        # Ensure proper UUID
                        if hasattr(item, 'object_id') and (not item.object_id or item.object_id == ""):
                            item.object_id = str(uuid.uuid4())
                        converted_items.append(item)
                    elif isinstance(item, tuple) and len(item) == 2:
                        # Convert tuple format (key, value) - this seems to be the LLM response format
                        key, value = item
                        # Skip tuples as they don't represent valid model instances
                        print(f"Skipping tuple format: {item}")
                        continue
                    elif isinstance(item, dict):
                        try:
                            # Remove any existing id from dict to let the model generate a new UUID
                            if 'object_id' in item:
                                del item['object_id']
                            converted_items.append(ModelCls(**item))
                        except Exception as e:
                            print(f"Failed to convert dict {item} to {ModelCls.__name__}: {e}")
                            continue
                    else:
                        # For emotions, we might get individual Emotion objects in the list
                        if ModelCls.__name__ == 'Emotion' and hasattr(item, 'content'):
                            # Ensure proper UUID
                            if hasattr(item, 'id') and (not item.object_id or item.object_id == ""):
                                item.object_id = str(uuid.uuid4())
                            converted_items.append(item)
                        else:
                            print(f"Unexpected item type for {ModelCls.__name__}: {type(item)} - {item}")
                return converted_items
            
            # Handle the case where we get raw model object
            if hasattr(data, 'object_id') and (not data.object_id or data.object_id == ""):
                data.object_id = str(uuid.uuid4())
            return data

        async def process_single_chunk(raw_chunk: str, stage_one: DiaryExtraction) -> Chunk:
            """chunk gets processed by parallel calls for every category flagged True"""
            tasks: dict[str, asyncio.Task] = {}

            # iterate over the dataclass fields dynamically
            for cat_key, category in stage_one.model_dump().items():
                flag = category["flag"]
                bullet_pts: list[str] = category["descriptions"]

                if not flag:
                    continue  # nothing to build for this category

                prompt, out_type = CAT_INFO[cat_key]
                full_prompt = build_prompt(cat_key, bullet_pts, raw_chunk)
                tasks[cat_key] = asyncio.create_task(
                    call_llm(full_prompt, out_type)
                )

            # wait for all tasks launched for this chunk
            results: dict[str, Any] = {}
            if tasks:
                completed = await asyncio.gather(*tasks.values())
                for cat_key, result in zip(tasks.keys(), completed):
                    results[cat_key] = result
            # Build Chunk object, defaulting to None when absent / failed
            # Convert LLM responses to correct pydantic models and provide required fields
            
            return Chunk(
                original_text = raw_chunk,
                summary       = [],  # Default empty summary
                connections   = None,  # Default no connections
                events        = convert_list(results.get("EventAction"), Event),
                people        = convert_list(results.get("People"), Person),
                thoughts      = convert_list(results.get("ThoughtReflections"), ThoughtReflection),
                emotions      = convert_list(results.get("Emotion"), Emotion),
                problems      = convert_list(results.get("Problem"), Problem),
                achievements  = convert_list(results.get("Achievement"), Achievement),
                goals         = convert_list(results.get("FutureIntentions"), FutureIntention),
            )

        async def stage_two_async() -> list[Chunk]:
            coroutines = [process_single_chunk(raw, s1) for raw, s1 in zip(raw_chunks, stage_one_results)]
            return await asyncio.gather(*coroutines)

        # execute stage_two_async directly since we're already in async context
        return await stage_two_async()




    def _cache_exists(self, cache_file: str) -> bool:
        return os.path.exists(cache_file)

    def _load_temp_cache(self, cache_file: str) -> list[Chunk]:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk(**e) for e in data]

    def _save_temp_cache(self, chunks: list[Chunk], cache_file: str) -> None:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f, indent=4, default=str, ensure_ascii=False)

    def _store_in_weaviate(self, chunks: list[Chunk]) -> None:
        try:
            with VectorDB() as db:
                # Clear existing collections before storing new data
                try:
                    # Handle different formats for collection list
                    collections = db.client.collections.list_all()

                    # Extract collection names regardless of the returned format
                    if isinstance(collections, list):
                        # Handle list format (older API versions)
                        if collections and hasattr(collections[0], 'name'):
                            collection_names = [c.name for c in collections]
                        else:
                            collection_names = [str(c) for c in collections]
                    elif isinstance(collections, dict):
                        # Handle dictionary format (newer API versions)
                        collection_names = list(collections.keys())
                    else:
                        # Fallback for other formats
                        collection_names = [str(collections)]

                    # Delete all collections
                    for collection_name in collection_names:
                        try:
                            db.client.collections.delete(collection_name)
                        except Exception as delete_err:
                            print(f"⚠️ Failed to delete collection {collection_name}: {delete_err}")
                except Exception as list_err:
                    print(f"⚠️ Failed to list collections: {list_err}")

                # Store chunks
                db.store_chunks(chunks)
        except Exception as e:
            print(f"❌ Failed to store in Weaviate: {e}")

    def _create_local_connections(self, chunks: list[Chunk]) -> list[Chunk]:
        graph_processor = GraphProcessor()
        for i, chunk in enumerate(chunks, 1):
            chunk.connections = graph_processor.create_local_graph_connections(chunk)
            print(f"[✓] Connections created for chunk {i}/{len(chunks)}")
        return chunks

    def _merge_with_global_graph(self, chunks: list[Chunk]) -> None:
        graph_processor = GraphProcessor()
        for i, chunk in enumerate(chunks, 1):
            graph_processor.merge_chunk_with_global_graph(chunk)
            print(f"[✓] Merged chunk {i}/{len(chunks)}")

    def encode(self, diary_text: str, cache_file: str = "data/processed_diary.json", batch_size: int | None = 10, store: bool = True) -> list[Chunk]:
        cache_path = Path(cache_file)
        cache_file_stage_two = str(cache_path.parent / f"{cache_path.stem}_stage_two.json")

        # Check if full cache exists
        if self._cache_exists(cache_file):
            print("[*] Loading full cached data...")
            chunks = self._load_temp_cache(cache_file)
            if store:
                print("[*] Storing cached data in Weaviate...")
                self._store_in_weaviate(chunks)
            print("[✓] Done.")
            return chunks

        # Check if stage two cache exists
        if self._cache_exists(cache_file_stage_two):
            print("[*] Loading stage two cached data...")
            chunks = self._load_temp_cache(cache_file_stage_two)

            print("[*] Creating connections...")
            chunks = self._create_local_connections(chunks)

            print("[*] Merging with global data..")
            self._merge_with_global_graph(chunks)

            print("[*] Saving full cache...")
            self._save_temp_cache(chunks, cache_file)

            if store:
                print("[*] Storing processed data in Weaviate...")
                self._store_in_weaviate(chunks)

            print("[✓] Done.")
            return chunks

        # No cache, start full process
        async def _run() -> list[Chunk]:
            async with LLM_OA("gpt-4.1-mini-2025-04-14") as llm:
                self.llm_oa = llm

                raw_chunks = self._diary_to_chunks(diary_text)



                print("[*] Starting Stage 1 - Flagging", batch_size)
                stage_one_results = await self._stage_one(raw_chunks, batch_size)
                print("[✓] Stage 1 completed.")

                print("[*] Starting Stage 2 - Creating Objects")
                chunks = await self._stage_two(raw_chunks, stage_one_results)
                self._save_temp_cache(chunks, cache_file_stage_two)
                print("[✓] Stage 2 completed.")

                print("[*] Starting Stage 3 - Creating Connections")
                chunks = self._create_local_connections(chunks)
                print("[✓] Stage 3 completed.")

                print("[*] Merging with global data (stage 5)...")
                self._merge_with_global_graph(chunks)
                print("[✓] Merging completed.")
                self._save_temp_cache(chunks, cache_file)
                if store:
                    print("[*] Storing processed data in Weaviate...")
                    self._store_in_weaviate(chunks)

                print("[✓] All stages complete.")
                return chunks

        return asyncio.run(_run())
