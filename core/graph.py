from typing import Optional
from difflib import SequenceMatcher
import weaviate.classes as wvc

from config.classes import *
from core.vector_db import VectorDB
from core.llm import LLM_OA

class GraphProcessor:
    """Handles all graph-related operations for diary processing."""
    
    def create_local_graph_connections(self, chunk: Chunk) -> list[Connection]:
        """Create connections between entities within a chunk."""
        def find_matching_people(name: str) -> list:
            name = name.lower()
            return [
                person for person in chunk.people or []
                if person.name and name in person.name.lower()
                or person.alias and name in person.alias.lower()
            ]

        def find_matching_emotions(title: str) -> list:
            title = title.lower()
            return [
                emotion for emotion in chunk.emotions or []
                if emotion.title and title == emotion.title.lower()
            ]

        def connect_entities(source_id, names: list[str], match_fn, conn_type: str):
            return [
                Connection(source_id=source_id, target_id=entity.object_id, type=conn_type)
                for name in names
                for entity in match_fn(name)
            ]

        connections = []

        # Events: participants → people
        for event in chunk.events or []:
            connections += connect_entities(event.object_id, event.participants or [], find_matching_people, "participated in")

        # Thoughts: people_mentioned & emotion
        for thought in chunk.thoughts or []:
            connections += connect_entities(thought.object_id, thought.people_mentioned or [], find_matching_people, "related to")
            if thought.emotion:
                connections += connect_entities(thought.object_id, [thought.emotion], find_matching_emotions, "related to")

        # Problems: people & emotions
        for problem in chunk.problems or []:
            connections += connect_entities(problem.object_id, problem.people or [], find_matching_people, "related to")
            connections += connect_entities(problem.object_id, problem.emotions or [], find_matching_emotions, "related to")

        # Achievements: people & emotions
        for achievement in chunk.achievements or []:
            connections += connect_entities(achievement.object_id, achievement.people or [], find_matching_people, "related to")
            connections += connect_entities(achievement.object_id, achievement.emotions or [], find_matching_emotions, "related to")

        # Goals: people
        for goal in chunk.goals or []:
            connections += connect_entities(goal.object_id, goal.people or [], find_matching_people, "related to")

        llm = LLM_OA("gpt-4.1-nano")
        # Generate connections using LLM for any remaining entities
        # Only show the original_text and the categories (events, people, thoughts, etc.) in the prompt. For each category, include the object_id, title, and description.

        prompt = f"""
        You are an expert in understanding relationships between entities generated out of a diary.
        Given the following objects, generate connections between entities based on the original text.
        Only generate connections that are not already present.

        Original Text: 
        '{chunk.original_text}'

        Present connections:
        {[(conn.source_id, conn.target_id, conn.type) for conn in chunk.connections or []]}

        Entities to connect:
        """
        for category, items in {
            "events": chunk.events,
            "people": chunk.people,
            "thoughts": chunk.thoughts,
            "problems": chunk.problems,
            "achievements": chunk.achievements,
            "goals": chunk.goals
        }.items():
            if items:
                prompt += f"\n{category.capitalize()}:\n"
                for item in items:
                    if hasattr(item, 'object_id') and hasattr(item, 'title') and hasattr(item, 'description'):
                        prompt += f"- ID: {item.object_id}, Title: {item.title}, Description: {item.description}\n"
                    elif hasattr(item, 'object_id') and hasattr(item, 'name'):
                        prompt += f"- ID: {item.object_id}, Name: {item.name}\n"
    

        llm_generated_connections = llm.generate_structured(prompt=prompt, desired_output_format=Connections).items
        connections += llm_generated_connections
        # Ensure all connnection source and target IDs are actual object IDs
        for conn in connections:
            if not isinstance(conn.source_id, str) or not isinstance(conn.target_id, str):
                print(f"Invalid connection found: {conn}")
                continue
            
            # Ensure source and target IDs are in the chunk's objects
            all_objects = (chunk.events or []) + (chunk.people or []) + (chunk.thoughts or []) + (chunk.problems or []) + (chunk.achievements or []) + (chunk.goals or [])
            if not any(obj.object_id == conn.source_id for obj in all_objects):
                print(f"Source ID {conn.source_id} not found in chunk objects. Removing connection.")
                connections.remove(conn)
                continue
            if not any(obj.object_id == conn.target_id for obj in all_objects):
                print(f"Target ID {conn.target_id} not found in chunk objects. Removing connection.")
                connections.remove(conn)
                continue

        return connections
    
    def merge_chunk_with_global_graph(self, chunk: Chunk) -> None:
        """
        Merges the category objects from a chunk into the global graph.
        """
        with VectorDB() as db:
            id_map = {}
            categories = ["events", "people", "thoughts", "problems", "achievements", "goals"] # Exclude emotions

            # Phase 1: Process all objects and collect ID mappings
            for category in categories:
                local_objects = getattr(chunk, category) or []
                for local_obj in local_objects:
                    global_obj = self.search_database(db, local_obj)
                    if self.check_for_match(local_obj, global_obj):
                        merged = self.merge_objects(local_obj, global_obj)
                        db.update_object(merged)
                        id_map[local_obj.object_id] = global_obj.object_id
            
            # Phase 2: Update all connections using complete ID mapping
            for connection in chunk.connections or []:
                if connection.source_id in id_map:
                    connection.source_id = id_map[connection.source_id]
                if connection.target_id in id_map:
                    connection.target_id = id_map[connection.target_id]
            
            # Phase 3: Clean up - remove merged objects and clear connector fields
            for category in categories:
                local_objects = getattr(chunk, category) or []
                # Remove objects that were merged
                setattr(chunk, category, [obj for obj in local_objects if obj.object_id not in id_map])
                
                # Clear connector fields from remaining local objects
                remaining_objects = getattr(chunk, category) or []
                for obj in remaining_objects:
                    if hasattr(obj, 'participants'):
                        obj.participants = []
                    if hasattr(obj, 'people_mentioned'):
                        obj.people_mentioned = []
                    if hasattr(obj, 'emotions'):
                        obj.emotions = []
                    if hasattr(obj, 'emotion'):
                        obj.emotion = ""
                    if hasattr(obj, 'people'):
                        obj.people = []

            # Store the updated chunk with global IDs
            db.store_chunks([chunk])
    
    def search_database(self, db: VectorDB, obj: BaseModel) -> Optional[BaseModel]:
        """Search for similar objects in the database using text and vector search."""
        config = getattr(obj.__class__, '_weaviate_config', None)
        if not config:
            raise ValueError(f"Model {obj.__class__.__name__} is not decorated with @weaviate_collection.")
        
        collection_name = config['collection_name']
        vector_fields = config.get('vectors', [])

        # Combined text and vector search with score fusion
        all_results: list[tuple[BaseModel, float]] = []

        # Text search on relevant fields
        text_fields = []
        search_query = ""
        
        if isinstance(obj, Event):
            text_fields = ["title"]
            search_query = obj.title
        elif isinstance(obj, Person):
            text_fields = ["name", "alias"]
            search_query = obj.name or ""
        elif isinstance(obj, ThoughtReflection):
            text_fields = ["title"]
            search_query = obj.title
        elif isinstance(obj, Problem):
            text_fields = ["title"]
            search_query = obj.title
        elif isinstance(obj, Achievement):
            text_fields = ["title"]
            search_query = obj.title
        elif isinstance(obj, FutureIntention):
            text_fields = ["title"]
            search_query = obj.title

        # Perform text search
        if search_query and text_fields:
            text_scores = db.text_search(
                collection_name=collection_name,
                field_names=text_fields,
                query=search_query,
                limit=5
            )
            # Weight text search scores (BM25 typically 0-10 range)
            for model, score in text_scores:
                all_results.append((model, score * 0.4))  # 40% weight for text

        # Perform vector search on content fields
        for vector_field in vector_fields:
            if hasattr(obj, vector_field):
                field_value = getattr(obj, vector_field)
                if field_value:
                    try:
                        collection = db.client.collections.get(collection_name).with_tenant(db.user_id)
                        query_embedding = db.embedder.embed_text(str(field_value))
                        
                        response = collection.query.near_vector(
                            near_vector=query_embedding,
                            target_vector=vector_field,
                            limit=5,
                            return_metadata=wvc.query.MetadataQuery(distance=True)
                        )
                        
                        # Convert distance to similarity score and add to combined results
                        for obj_result in response.objects:
                            # Convert properties to BaseModel
                            from core.schema_generator import get_model_class
                            model_class = get_model_class(collection_name)
                            if model_class:
                                # Sanitize properties for model creation
                                properties = obj_result.properties.copy()
                                # Fix list fields - ensure they're lists, not empty strings
                                list_fields = ['participants', 'people_mentioned', 'people', 'emotions']
                                for field in list_fields:
                                    if field in properties and properties[field] == '':
                                        properties[field] = []
                                # Preserve the actual UUID from Weaviate
                                properties['object_id'] = str(obj_result.uuid)
                                model_instance = model_class(**properties)
                                # Convert distance to similarity (closer = higher score)
                                similarity_score = 1 / (1 + obj_result.metadata.distance)
                                # Weight vector search scores (60% weight)
                                all_results.append((model_instance, similarity_score * 0.6))
                                    
                    except Exception as e:
                        print(f"Vector search failed for {vector_field}: {e}")

        # Return the best match based on combined scores
        if all_results:
            # Find the result with the highest score
            best_result = max(all_results, key=lambda x: x[1])
            return best_result[0]  # Return just the model, not the tuple
        else:
            return None
    

    def check_for_match(self, local_object: BaseModel, global_object: BaseModel) -> bool:
        """
        Checks if a given object matches an existing object in the global graph.
        Uses multiple similarity measures with configurable thresholds.
        """
        if not global_object or not isinstance(local_object, global_object.__class__):
            return False
        
        # Thresholds for different match types
        EXACT_MATCH_THRESHOLD = 0.95    # Very high confidence
        FUZZY_MATCH_THRESHOLD = 0.8     # Good confidence  
        SEMANTIC_MATCH_THRESHOLD = 0.7   # Moderate confidence

        def fuzzy_similarity(str1: str, str2: str) -> float:
            """Calculate fuzzy string similarity (0-1)"""
            if not str1 or not str2:
                return 0.0
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

        # 1. Exact title match (highest priority)
        if hasattr(local_object, 'title') and hasattr(global_object, 'title'):
            if local_object.title and global_object.title:
                if local_object.title.lower() == global_object.title.lower():
                    return True
                
                # Fuzzy title match
                title_similarity = fuzzy_similarity(local_object.title, global_object.title)
                if title_similarity >= EXACT_MATCH_THRESHOLD:
                    return True

        # 2. Person-specific matching (name + alias)
        if isinstance(local_object, Person):
            # Name matching
            if local_object.name and global_object.name:
                name_similarity = fuzzy_similarity(local_object.name, global_object.name)
                if name_similarity >= FUZZY_MATCH_THRESHOLD:
                    return True
            
            # Cross-reference name with alias
            if local_object.name and global_object.alias:
                if local_object.name.lower() in global_object.alias.lower():
                    return True
            if local_object.alias and global_object.name:
                if global_object.name.lower() in local_object.alias.lower():
                    return True

        # 3. Content-based semantic similarity for objects with descriptions
        if hasattr(local_object, 'description') and hasattr(global_object, 'description'):
            if local_object.description and global_object.description:
                desc_similarity = fuzzy_similarity(local_object.description, global_object.description)
                if desc_similarity >= SEMANTIC_MATCH_THRESHOLD:
                    return True

        # 4. Event-specific location + time matching
        if isinstance(local_object, Event):
            location_match = False
            time_match = False
            
            if local_object.location and global_object.location:
                location_similarity = fuzzy_similarity(local_object.location, global_object.location)
                location_match = location_similarity >= FUZZY_MATCH_THRESHOLD
            
            if local_object.time and global_object.time:
                time_similarity = fuzzy_similarity(local_object.time, global_object.time)
                time_match = time_similarity >= FUZZY_MATCH_THRESHOLD
                
            # If both location and time have good matches, consider it a match
            if location_match and time_match:
                return True

        # 5. Multi-field fuzzy matching for high-confidence cases
        total_score = 0.0
        field_count = 0
        
        # Score title similarity
        if hasattr(local_object, 'title') and hasattr(global_object, 'title'):
            if local_object.title and global_object.title:
                total_score += fuzzy_similarity(local_object.title, global_object.title) * 2  # Weight titles higher
                field_count += 2

        # Score description similarity  
        if hasattr(local_object, 'description') and hasattr(global_object, 'description'):
            if local_object.description and global_object.description:
                total_score += fuzzy_similarity(local_object.description, global_object.description)
                field_count += 1

        # Calculate average weighted score
        if field_count > 0:
            average_score = total_score / field_count
            if average_score >= FUZZY_MATCH_THRESHOLD:
                return True

        return False
    
    def merge_objects(self, local_object: BaseModel, global_object: BaseModel) -> BaseModel:
        """
        Merges a local object with a global object.
        Returns the merged object.
        """
        # Make sure the local object is the same type as the global object
        if not isinstance(local_object, global_object.__class__):
            print("Type mismatch during merge!")
            return global_object
        
        # Events: local description added before global description. Local location added after global location, separated by a /. Rest take global values.
        if isinstance(local_object, Event):
            merged_description = f"{local_object.description} \n {global_object.description}"
            merged_location = f"{global_object.location} / {local_object.location}" if local_object.location else global_object.location
            return Event(
                object_id=global_object.object_id,
                title=global_object.title,
                description=merged_description,
                location=merged_location,
                time=global_object.time,
                participants=global_object.participants or local_object.participants
            )
        
        # People: local alias added before global alias. Separated by /. The rest separated by \n. Local description added before global description. Local relationship_to_user added before global relationship_to_user. Rest take global values.
        elif isinstance(local_object, Person):
            merged_alias = f"{local_object.alias} / {global_object.alias}" if local_object.alias else global_object.alias
            merged_description = f"{local_object.description} \n {global_object.description}"
            merged_relationship = f"{local_object.relationship_to_user} / {global_object.relationship_to_user}" if local_object.relationship_to_user else global_object.relationship_to_user
            return Person(
                object_id=global_object.object_id,
                name=global_object.name,
                alias=merged_alias,
                description=merged_description,
                relationship_to_user=merged_relationship
            )
        
        # ThoughtReflection: local description added before global description. Rest take global values.
        elif isinstance(local_object, ThoughtReflection):
            merged_description = f"{local_object.description} \n {global_object.description}"
            return ThoughtReflection(
                object_id=global_object.object_id,
                title=global_object.title,
                description=merged_description,
                people_mentioned=global_object.people_mentioned or local_object.people_mentioned,
                emotion=global_object.emotion or local_object.emotion
            )
        
        # Problem: local description added before global description. Rest take global values.
        elif isinstance(local_object, Problem):
            merged_description = f"{local_object.description} \n {global_object.description}"
            return Problem(
                object_id=global_object.object_id,
                title=global_object.title,
                description=merged_description,
                people=global_object.people or local_object.people,
                emotions=global_object.emotions or local_object.emotions
            )
        
        # Achievement: local description added before global description. Rest take global values.
        elif isinstance(local_object, Achievement):
            merged_description = f"{local_object.description} \n {global_object.description}"
            return Achievement(
                object_id=global_object.object_id,
                title=global_object.title,
                description=merged_description,
                people=global_object.people or local_object.people,
                emotions=global_object.emotions or local_object.emotions
            )

        # FutureIntention: local description added before global description. Rest take global values.
        elif isinstance(local_object, FutureIntention):
            merged_description = f"{local_object.description} \n {global_object.description}"
            return FutureIntention(
                object_id=global_object.object_id,
                title=global_object.title,
                description=merged_description,
                people=global_object.people or local_object.people
            )

        else:
            print(f"Unknown object type for merging: {local_object.__class__.__name__}")
            return global_object