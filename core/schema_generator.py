from typing import Dict, List, Type, Any, get_origin, get_args
from pydantic import BaseModel
import weaviate.classes as wvc
from datetime import datetime
import inspect


_registered_collections: Dict[str, Type[BaseModel]] = {}


def weaviate_collection(
    name: str = None,
    vectors: List[str] = None,
    multi_tenant: bool = True,
    subcollections: Dict[str, str] = None,
    parent_id_field: str = None
):
    """
    Decorator to mark a Pydantic model as a Weaviate collection.
    
    Args:
        name: Collection name (required - must be explicitly specified)
        vectors: List of field names that should have vector embeddings
        multi_tenant: Enable multi-tenancy (default: True)
        subcollections: Dict mapping attribute names to collection names (for main Chunk class)
        parent_id_field: Field name to use for linking to parent records (e.g., "id")
    """
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        # Collection name is required
        if not name:
            raise ValueError(f"Collection name must be explicitly specified for {cls.__name__}")
        
        collection_name = name
        
        # Store metadata on the class
        cls._weaviate_config = {
            'collection_name': collection_name,
            'vectors': vectors or [],
            'multi_tenant': multi_tenant,
            'model_class': cls,
            'subcollections': subcollections or {},
            'parent_id_field': parent_id_field
        }
        
        # Register the collection globally
        _registered_collections[collection_name] = cls
        
        return cls
    
    return decorator


def get_weaviate_data_type(python_type: Type) -> wvc.config.DataType:
    """Map Python types to Weaviate DataTypes."""
    # Handle Optional/Union types
    origin = get_origin(python_type)
    if origin is not None:
        args = get_args(python_type)
        # For Optional[T] which is Union[T, None], get the non-None type
        if origin is type(None) or (hasattr(origin, '__name__') and 'Union' in str(origin)):
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                python_type = non_none_types[0]
    
    # Basic type mapping
    type_mapping = {
        str: wvc.config.DataType.TEXT,
        int: wvc.config.DataType.INT,
        float: wvc.config.DataType.NUMBER,
        bool: wvc.config.DataType.BOOL,
        datetime: wvc.config.DataType.DATE,
    }
    
    return type_mapping.get(python_type, wvc.config.DataType.TEXT)


def generate_properties_from_model(model_class: Type[BaseModel]) -> List[wvc.config.Property]:
    """Generate Weaviate properties from Pydantic model fields."""
    properties = []
    config = getattr(model_class, '_weaviate_config', {})
    parent_id_field = config.get('parent_id_field')
    has_parent_field = False
    
    for field_name, field_info in model_class.model_fields.items():
        # Get the field type
        field_type = field_info.annotation
        
        # Convert to Weaviate DataType
        weaviate_type = get_weaviate_data_type(field_type)
        
        # Create property
        properties.append(
            wvc.config.Property(name=field_name, data_type=weaviate_type)
        )
        
        if parent_id_field and field_name == parent_id_field:
            has_parent_field = True
    
    # Add parent_id_field for linking if specified and not already present
    if parent_id_field and not has_parent_field:
        properties.append(
            wvc.config.Property(name=parent_id_field, data_type=wvc.config.DataType.TEXT)
        )
    
    return properties


def generate_vector_config(vectors: List[str]) -> Dict[str, Any]:
    """Generate vector configuration for specified fields."""
    if not vectors:
        return {}

    # Create default vectorizer config
    result = {
        "default": wvc.config.Configure.vectorizer.none(
            vector_dimensions=1536  # OpenAI embedding dimension
        )
    }

    # Add named vectors for each vector field
    named_vectors = {}
    for vector_field in vectors:
        named_vectors[vector_field] = wvc.config.Configure.vector.named_vectors(
            source_vectors=None,  # We'll provide vectors directly on insert
            vector_dimensions=1536  # OpenAI embedding dimension
        )

    if named_vectors:
        result["vectors"] = named_vectors

    return result


def generate_collection_config(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Generate complete Weaviate collection configuration from Pydantic model."""
    if not hasattr(model_class, '_weaviate_config'):
        raise ValueError(f"Model {model_class.__name__} is not decorated with @weaviate_collection")
    
    config = model_class._weaviate_config
    
    collection_config = {
        'name': config['collection_name'],
        'properties': generate_properties_from_model(model_class),
        'vector_config': generate_vector_config(config['vectors'])
    }
    
    # Add multi-tenancy if enabled
    if config['multi_tenant']:
        collection_config['multi_tenancy_config'] = wvc.config.Configure.multi_tenancy(enabled=True)
    
    return collection_config


def discover_collections_in_module(module) -> Dict[str, Type[BaseModel]]:
    """
    Discover all @weaviate_collection decorated models in a module.
    
    Args:
        module: Python module to scan
        
    Returns:
        Dict mapping collection names to model classes
    """
    collections = {}
    
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseModel) and 
            hasattr(obj, '_weaviate_config')):
            config = obj._weaviate_config
            collections[config['collection_name']] = obj
    
    return collections


def get_model_class(collection_name: str) -> Type[BaseModel] | None:
    """Return the Pydantic model class registered for a collection name."""
    return _registered_collections.get(collection_name)