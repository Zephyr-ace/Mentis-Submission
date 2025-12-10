import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
import os
from core.embedder import TextEmbedder
from core.schema_generator import discover_collections_in_module, generate_vector_config, generate_properties_from_model, get_model_class
import config.classes as classes
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True) #.env variables



class VectorDB:
    def __init__(self, user_id= None, cluster_url=None, api_key=None):
        if user_id is None:
            user_id = os.getenv("USER_ID")
        if not user_id:
            raise EnvironmentError("VectorDB requires a user_id. "
                                   "Please pass one in or set the USER_ID environment variable.")
        self.user_id = user_id
        self.embedder = TextEmbedder()
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url or os.getenv("WEAVIATE_URL"),
            auth_credentials=wvc.init.Auth.api_key(
                api_key or os.getenv("WEAVIATE_API_KEY")
            ),
            # everything below replaces the old `timeout=` argument
            additional_config=AdditionalConfig(
                # You can pass a tuple (connect, read) **or** a Timeout object
                timeout=Timeout(
                    query=60,  # read / GraphQL / generate ops
                    insert=90,  # batch & inserts
                    init=5  # start‑up health check
                )
            )
        )
        self._create_schema()
        self._create_tenant()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()
    
    def _create_schema(self):
        """Create Weaviate collections from decorated Pydantic models."""
        collections = self.client.collections
        
        # Discover all collections from the classes module
        discovered_collections = discover_collections_in_module(classes)
        
        # Create each collection if it doesn't exist
        for collection_name, model_class in discovered_collections.items():
            if not collections.exists(collection_name):
                try:
                    config = model_class._weaviate_config
                    
                    # Build collection creation arguments
                    create_args = {
                        'name': collection_name,
                        'properties': generate_properties_from_model(model_class)
                    }
                    
                    # Only add vector_config if there are vectors
                    vector_configs = generate_vector_config(config['vectors'])
                    if vector_configs:
                        create_args['vector_config'] = vector_configs
                    
                    # Add multi-tenancy if enabled
                    if config.get('multi_tenant', True):
                        create_args['multi_tenancy_config'] = (
                            wvc.config.Configure.multi_tenancy(enabled=True)
                        )
                    
                    collections.create(**create_args)
                    
                except Exception as e:
                    print(f"❌ Failed to create collection {collection_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    
    def _extract_model_properties(self, instance, parent_id=None, subcollections_to_exclude=None):
        """Extract properties from a Pydantic model instance for Weaviate storage."""
        from datetime import datetime
        
        if subcollections_to_exclude is None:
            subcollections_to_exclude = {}
        
        properties = {}
        config = instance._weaviate_config
        parent_id_field = config.get('parent_id_field')
        
        # Extract all field values from the instance
        for field_name, field_value in instance.model_dump().items():
            # Skip subcollection fields - these are stored in separate collections
            if field_name in subcollections_to_exclude:
                continue
            
            # Convert values to appropriate formats for Weaviate
            if isinstance(field_value, datetime):
                properties[field_name] = field_value.strftime("%Y-%m-%dT%H:%M:%SZ")
            elif field_value is None:
                properties[field_name] = ""
            elif isinstance(field_value, (list, dict)):
                # These should be handled via subcollections, but if they appear here,
                # convert to string as fallback since Weaviate schema expects strings
                if isinstance(field_value, list):
                    # Convert list to comma-separated string, or empty string if empty
                    properties[field_name] = ", ".join(str(item) for item in field_value) if field_value else ""
                else:
                    properties[field_name] = str(field_value) if field_value else ""
            else:
                # Simple types (str, int, bool) are stored directly
                properties[field_name] = field_value
        
        # Add parent linking field if configured and parent_id is provided
        if parent_id_field and parent_id and not hasattr(instance, parent_id_field):
            properties[parent_id_field] = parent_id
        
        return properties
    


    def _create_tenant(self):
        """Create tenant for this user if it doesn't exist"""
        collections = self.client.collections

        # Get all registered collection names dynamically
        discovered_collections = discover_collections_in_module(classes)

        for collection_name in discovered_collections.keys():
            try:
                collection = collections.get(collection_name)

                # Check if tenant exists
                existing_tenants = collection.tenants.get()
                tenant_names = []

                # Handle different return formats from Weaviate API
                if isinstance(existing_tenants, dict):
                    # Dictionary format (newer API versions)
                    tenant_names = list(existing_tenants.keys())
                elif isinstance(existing_tenants, list):
                    # List format (older API versions)
                    for tenant in existing_tenants:
                        if isinstance(tenant, str):
                            tenant_names.append(tenant)
                        elif hasattr(tenant, 'name'):
                            tenant_names.append(tenant.name)
                        else:
                            # Fallback for unexpected formats
                            tenant_names.append(str(tenant))

                # Create tenant if it doesn't exist
                if self.user_id not in tenant_names:
                    collection.tenants.create([wvc.tenants.Tenant(name=self.user_id)])
            except Exception as e:
                print(f"❌ Failed to create tenant for {collection_name}: {e}")

    def store_chunks(self, chunks):
        """Store chunks and their subcategories using batch operations for better performance."""

        # Group all data by collection for batch processing
        collection_data = {}  # {collection_name: [(instance, parent_id), ...]}

        for chunk in chunks:
            try:
                # Add main chunk
                chunk_config = chunk._weaviate_config
                chunk_collection = chunk_config['collection_name']

                if chunk_collection not in collection_data:
                    collection_data[chunk_collection] = []
                collection_data[chunk_collection].append((chunk, None))

                # Add subcollection items
                subcollections = chunk_config.get('subcollections', {})
                for attr_name, collection_name in subcollections.items():
                    items = getattr(chunk, attr_name, []) or []

                    # Skip if items is not a list
                    if isinstance(items, str):
                        continue

                    if collection_name not in collection_data:
                        collection_data[collection_name] = []

                    for item in items:
                        if hasattr(item, '_weaviate_config'):
                            collection_data[collection_name].append((item, chunk.chunk_id))
            except Exception as e:
                print(f"⚠️ Error processing chunk: {e}")

        # Batch process each collection
        for collection_name, instances_data in collection_data.items():
            if instances_data:
                try:
                    self._batch_store_instances(collection_name, instances_data)
                except Exception as e:
                    print(f"⚠️ Error batch storing data for collection {collection_name}: {e}")
    
    def _batch_store_instances(self, collection_name: str, instances_data: list):
        """Batch store multiple instances in a single collection."""
        try:
            collection = self.client.collections.get(collection_name).with_tenant(self.user_id)

            # Prepare all data for batch insert
            batch_objects = []
            texts_to_embed = {}  # {unique_key: text} for batch embedding
            embedding_map = {}   # {unique_key: (instance_idx, field_name)}

            for idx, (instance, parent_id) in enumerate(instances_data):
                try:
                    config = instance._weaviate_config

                    # Extract properties
                    properties = self._extract_model_properties(instance, parent_id, config.get('subcollections', {}))

                    # Collect texts that need embedding
                    vector_fields = config.get('vectors', [])
                    instance_embeddings = {}

                    for field_name in vector_fields:
                        field_value = getattr(instance, field_name, None)
                        if field_value is not None:
                            text_value = str(field_value).strip()
                            if text_value:  # Only embed non-empty content
                                # Create unique key for this text
                                unique_key = f"{idx}_{field_name}"
                                texts_to_embed[unique_key] = text_value
                                embedding_map[unique_key] = (idx, field_name)

                    batch_objects.append({
                        'properties': properties,
                        'vectors': instance_embeddings  # Will be filled after batch embedding
                    })
                except Exception as e:
                    print(f"⚠️ Error preparing instance for collection {collection_name}: {e}")

            # Batch generate embeddings if needed
            if texts_to_embed:
                try:
                    embeddings = self.embedder.embed_text_dict(texts_to_embed)

                    # Map embeddings back to batch objects
                    for unique_key, embedding in embeddings.items():
                        try:
                            instance_idx, field_name = embedding_map[unique_key]
                            if 'vectors' not in batch_objects[instance_idx] or batch_objects[instance_idx]['vectors'] is None:
                                batch_objects[instance_idx]['vectors'] = {}
                            batch_objects[instance_idx]['vectors'][field_name] = embedding
                        except Exception as e:
                            print(f"⚠️ Error mapping embedding: {e}")
                except Exception as e:
                    print(f"⚠️ Error generating embeddings: {e}")

            # Batch insert all objects
            with collection.batch.dynamic() as batch:
                for obj in batch_objects:
                    try:
                        batch.add_object(
                            properties=obj['properties'],
                            vector=obj['vectors'] if obj.get('vectors') else None
                        )
                    except Exception as e:
                        print(f"⚠️ Error adding object to batch: {e}")

            # Check for any errors
            if hasattr(collection.batch, 'failed_objects') and collection.batch.failed_objects:
                print(f"⚠️ {len(collection.batch.failed_objects)} objects failed to insert")
                for failed_obj in collection.batch.failed_objects[:5]:  # Show first 5 errors
                    if hasattr(failed_obj, 'message'):
                        print(f"⚠️ Error: {failed_obj.message}")
                    else:
                        print(f"⚠️ Error with failed object")

        except Exception as e:
            print(f"❌ Error in batch store for collection {collection_name}: {e}")
    

    def delete_user_data(self, delete: bool = True):
        """Delete entire collections/schema (all users' data will be lost)"""
        if delete:
            try:
                with VectorDB(user_id=self.user_id) as db:
                    collections = db.client.collections
                    discovered_collections = discover_collections_in_module(classes)
                    
                    for collection_name in discovered_collections.keys():
                        if collections.exists(collection_name):
                            collections.delete(collection_name)
            except Exception as e:
                print(f"❌ Failed to delete schema: {e}")

    def vector_search(self, collection_name, query, vector_name, limit=10):
        """Generic vector search method"""
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
        query_embedding = self.embedder.embed_text(query)
        
        response = collection.query.near_vector(
            near_vector=query_embedding,
            target_vector=vector_name,
            limit=limit
        )
        return [obj.properties for obj in response.objects]
    
    
    def text_search(self, collection_name, query, limit=10) -> list:
        """Generic text search method using BM25 search"""
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
        
        # Use BM25 search for text search
        response = collection.query.bm25(
            query=query,
            limit=limit
        )
        return [obj.properties for obj in response.objects]
    
    
    def text_search(self, collection_name: str, field_names: list[str], query: str, limit=10) -> list[tuple[BaseModel, float]]:
        """Generic text search method using BM25 search on specific fields.
        Returns a list of tuples (model, score).
        """
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
        
        # Use BM25 search for text search on specific fields (field_names)
        response = collection.query.bm25(
            query=query,
            query_properties=field_names,
            return_metadata=wvc.query.MetadataQuery(score=True),
            limit=limit
        )

        # Convert properties to BaseModel instances and build result list
        results = []
        for obj in response.objects:
            model_class = get_model_class(collection_name)
            if model_class:
                # Sanitize data before creating model instance
                properties = obj.properties.copy()
                # Fix list fields - ensure they're lists, not strings
                list_fields = ['participants', 'people_mentioned', 'people', 'emotions']
                for field in list_fields:
                    if field in properties:
                        if properties[field] == '':
                            properties[field] = []
                        elif isinstance(properties[field], str):
                            # Convert comma-separated string to list
                            properties[field] = [item.strip() for item in properties[field].split(',') if item.strip()]
                model_instance = model_class(**properties)
                results.append((model_instance, obj.metadata.score))
        
        return results

    def hybrid_search(self, collection_name: str, query: str, vector_field: str, text_fields: list[str], alpha: float = 0.5, limit: int = 8, min_score: float = 0.0) -> list[tuple[BaseModel, float]]:
        """Weaviate's built-in hybrid search combining vector similarity and BM25 text search"""
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
        query_embedding = self.embedder.embed_text(query)
        
        response = collection.query.hybrid(
            query=query,
            target_vector=vector_field,
            query_properties=text_fields,
            alpha=alpha,  # 0.0 = pure BM25, 1.0 = pure vector
            return_metadata=wvc.query.MetadataQuery(score=True),
            vector={vector_field: query_embedding},  # Provide manual vector for self_provided
            limit=limit
        )
        
        results = []
        for obj in response.objects:
            # Filter by minimum score threshold
            if obj.metadata.score >= min_score:
                model_class = get_model_class(collection_name)
                if model_class:
                    properties = obj.properties.copy()
                    # Fix list fields
                    list_fields = ['participants', 'people_mentioned', 'people', 'emotions']
                    for field in list_fields:
                        if field in properties:
                            if properties[field] == '':
                                properties[field] = []
                            elif isinstance(properties[field], str):
                                properties[field] = [item.strip() for item in properties[field].split(',') if item.strip()]
                    model_instance = model_class(**properties)
                    results.append((model_instance, obj.metadata.score))
        
        return results

    def parallel_hybrid_search(self, query_classification_pairs: list[tuple[str, str]], min_score: float = 0.3) -> dict[str, list[tuple[BaseModel, float]]]:
        """Search collections in parallel for each query-classification pair"""
        import concurrent.futures
        
        # Alpha factors by classification
        alpha_factors = {
            'Event': 0.5,
            'Person': 0.3, 
            'ThoughtReflection': 0.7,
            'Emotion': 0.6,
            'Problem': 0.6,
            'Achievement': 0.5,
            'FutureIntention': 0.6
        }
        
        # Collection mapping
        collection_mapping = {
            'Event': 'ChunkEvent',
            'Person': 'ChunkPerson',
            'ThoughtReflection': 'ChunkThought',
            'Emotion': 'ChunkEmotion',
            'Problem': 'ChunkProblem',
            'Achievement': 'ChunkAchievement',
            'FutureIntention': 'ChunkFutureIntention'
        }
        
        # Vector and text field mapping
        field_mapping = {
            'ChunkEvent': {'vector': 'title', 'text': ['title', 'description']},
            'ChunkPerson': {'vector': 'name', 'text': ['name', 'description']},
            'ChunkThought': {'vector': 'title', 'text': ['title', 'description']},
            'ChunkEmotion': {'vector': 'title', 'text': ['title', 'description']},
            'ChunkProblem': {'vector': 'title', 'text': ['title', 'description']},
            'ChunkAchievement': {'vector': 'title', 'text': ['title', 'description']},
            'ChunkFutureIntention': {'vector': 'title', 'text': ['title', 'description']}
        }
        
        def search_single(query: str, query_category: str) -> tuple[str, list]:
            if query_category not in collection_mapping:
                print(f"Error: Unrecognized query category '{query_category}'")
                return query_category, []
            
            collection_name = collection_mapping[query_category]
            alpha = alpha_factors[query_category]
            fields = field_mapping[collection_name]
            
            try:
                results = self.hybrid_search(
                    collection_name=collection_name,
                    query=query,
                    vector_field=fields['vector'],
                    text_fields=fields['text'],
                    alpha=alpha,
                    limit=8,
                    min_score=min_score
                )
                return query_category, results
            except Exception as e:
                print(f"Error searching {collection_name}: {e}")
                return query_category, []
        
        # Execute searches in parallel
        results_by_collection = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(query_classification_pairs)) as executor:
            future_to_category = {
                executor.submit(search_single, query, query_category): query_category 
                for query, query_category in query_classification_pairs
            }
            
            for future in concurrent.futures.as_completed(future_to_category):
                category, results = future.result()
                if category not in results_by_collection:
                    results_by_collection[category] = []
                results_by_collection[category].extend(results)
        
        return results_by_collection


    def update_object(self, input_model):
        """Update an object in the database by its ID, with fallback to insert if object doesn't exist"""
        # Get the collection name from the model's weaviate config
        config = input_model._weaviate_config
        collection_name = config['collection_name']
        
        # Get the collection with tenant
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
        
        # Extract properties from the input model
        properties = self._extract_model_properties(input_model, subcollections_to_exclude=config.get('subcollections', {}))
        
        # Generate embeddings for vector fields if they exist
        vector_fields = config.get('vectors', [])
        vectors = {}
        
        for field_name in vector_fields:
            field_value = getattr(input_model, field_name, None)
            if field_value is not None:
                text_value = str(field_value).strip()
                if text_value:  # Only embed non-empty content
                    vectors[field_name] = self.embedder.embed_text(text_value)
        
        # First try to check if object exists
        try:
            existing_obj = collection.query.fetch_object_by_id(input_model.object_id)
            if existing_obj:
                # Object exists, update it
                collection.data.update(
                    uuid=input_model.object_id,
                    properties=properties,
                    vector=vectors if vectors else None
                )
                return True
        except Exception:
            # Object doesn't exist or query failed, fall through to insert
            pass
        
        # Object doesn't exist, insert it instead
        try:
            collection.data.insert(
                uuid=input_model.object_id,
                properties=properties,
                vector=vectors if vectors else None
            )
            return True
        except Exception as e:
            print(f"❌ Failed to upsert object {input_model.object_id} in {collection_name}: {e}")
            return False
        
    def get_all_objects(self, collection_name: str) -> list[BaseModel]:
        """Retrieve all objects from a collection as a list of Pydantic models."""
        collection = self.client.collections.get(collection_name).with_tenant(self.user_id)

        objects = []
        for item in collection.iterator():
            # Create properties dict with UUID preserved
            properties = item.properties.copy()
            
            object = self.properties_to_base_model(collection_name, properties)
            if object:
                objects.append(object)

        return objects
    
    def properties_to_base_model(self, collection_name: str, properties: dict[str, any]) -> BaseModel:
        """
        Convert raw Weaviate properties dict to a Pydantic model instance,
        fixing list fields that are empty strings.
        """
        def sanitize_properties(model_cls: type[BaseModel], props: dict[str, any]) -> dict[str, any]:
            for field_name, field in model_cls.model_fields.items():
                # Check if the field is some kind of list
                origin = getattr(field.annotation, '__origin__', None)
                if origin is list:
                    # Fix empty string to empty list
                    if isinstance(props.get(field_name), str) and props[field_name] == "":
                        props[field_name] = []
            return props
        
        model_class = get_model_class(collection_name)
        sanitized_props = sanitize_properties(model_class, properties.copy())
        return model_class(**sanitized_props)

    def get_connected_objects(self, object_ids: list[str]) -> list[BaseModel]:
        """
        Get all objects connected to the given object IDs.
        
        Args:
            object_ids: List of object IDs to find connections for
            
        Returns:
            List of Objects connected to the given IDs
        """
        
        if not object_ids:
            return []
            
        # First, find all connections involving the given object_ids
        connection_collection = self.client.collections.get("Connection").with_tenant(self.user_id)
        all_connections_response = connection_collection.query.fetch_objects(limit=10000)
        
        # Collect all connected object IDs
        connected_object_ids = set()
        
        for obj in all_connections_response.objects:
            properties = obj.properties
            source_id = properties.get('source_id')
            target_id = properties.get('target_id')
            
            # If source_id is in our input list, add target_id to connected objects
            if source_id in object_ids:
                connected_object_ids.add(target_id)
            # If target_id is in our input list, add source_id to connected objects  
            if target_id in object_ids:
                connected_object_ids.add(source_id)
        
        # Now fetch the actual objects from all collections
        results = []
        seen_object_ids = set()
        
        # Get all collection names from the discovered collections
        from core.schema_generator import discover_collections_in_module
        import config.classes as classes
        collection_configs = discover_collections_in_module(classes)
        
        for collection_name in collection_configs.keys():
            if collection_name == "Connection":  # Skip connections themselves
                continue
                
            try:
                collection = self.client.collections.get(collection_name).with_tenant(self.user_id)
                collection_objects = collection.query.fetch_objects(limit=10000)
                
                for obj in collection_objects.objects:
                    obj_id = obj.properties.get('object_id')
                    if obj_id and obj_id in connected_object_ids and obj_id not in seen_object_ids:
                        seen_object_ids.add(obj_id)
                        
                        # Convert to BaseModel
                        connected_object = self.properties_to_base_model(collection_name, obj.properties.copy())
                        results.append(connected_object)
                        
            except Exception as e:
                # Skip collections that don't exist or have issues
                continue
        
        return results