from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, conint, field_validator
from datetime import datetime
import uuid
from core.schema_generator import weaviate_collection



# SimpleRag
@weaviate_collection(
    name="simple_rag",
    vectors=["content"],
    multi_tenant=True
)
class SimpleRagChunk(BaseModel):
    content: str
    chunk_index: int

# SummaryRag
@weaviate_collection(
    name="Summary",
    vectors=["content"],
    multi_tenant=True
)
class ChunkSummary(BaseModel):
    content: str
    chunk_index: int


# check
class TrueFalse(BaseModel):
    answer: bool


# evaluation
class OutputEval(BaseModel):
    relevance: conint(ge=0, le=2) = Field(
        ...,
        description="Relevance score: 0 = none, 1 = partial, 2 = highly relevant",
        alias="Relevance"
    )
    overallUtility: conint(ge=0, le=2) = Field(
        ...,
        description="Utility score: 0 = no value, 1 = nice-to-have, 2 = crucial",
        alias="Overallutility"
    )


#retriever

class QueryRewriteItem(BaseModel):
    rewritten_query: str = Field(..., description="Rewritten query")
    query_category: str = Field(..., description="Query category")

class QueriesAndClassification(BaseModel):
    items: list[QueryRewriteItem] = Field(..., description="Queries with their categories")


class FilteredResults(BaseModel):
    relevant_results: list[str] = Field(..., description="Relevant results")


# Encoder

# stage 1
class Category(BaseModel):
    flag: bool = Field(
        ...,
        description="True if at least one object for this category must be constructed.")
    descriptions: List[str] = Field(
        ...,
        description="1-5 bullet points (≤ 5 words) that state which diary details belong here. Empty if flag is false."
    )


# categories for information objects
class DiaryExtraction(BaseModel):
    EventAction: Category = Field(..., alias="Events & Actions")
    People: Category = Field(..., alias="People")
    ThoughtReflections: Category = Field(..., alias="Thoughts & Reflections")
    Emotion: Category = Field(..., alias="Feelings & Emotions")
    Problem  : Category = Field(..., alias="Problems & Concerns")
    Achievement: Category = Field(..., alias="Achievements")
    FutureIntentions: Category = Field(..., alias="Future Intentions")


# stage 2

# Events
@weaviate_collection(name="ChunkEvent", vectors=["title", "content"], parent_id_field="chunk_id")
class Event(BaseModel):
    """
    One single event extracted from a diary entry.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")

    description: str = Field(..., description="what actually happened")
    title: str = Field(..., description="title for description")
    time: str = Field(..., description="explicit date/time or approximate (e.g. 'morning')")
    location: str = Field(..., description="physical or virtual place")
    participants: List[str] = Field(..., description="people that attended")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class Events(BaseModel):
    """
    The model will always return exactly this object: a list of events.
    """
    items: List[Event] = Field(..., description="One item for every description supplied by the caller")


# People
# @weaviate_collection(name="PersonRelation", vectors=["comment"], parent_id_field="person_id")
# class RelationToOther(BaseModel):
#     """
#     One explicit relationship this person has with another person that is
#     mentioned in the same diary entry.
#     """
#     object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this relationship")
#     person_id: str = Field(..., description="ID of the person who has this relationship")
#     other_person: str = Field(
#         ...,
#         description="Name or identifier of the other person."
#     )
#     comment: Optional[str] = Field(
#         default=None,
#         description="Short quote or very brief paraphrase that evidences the relationship."
#     )
#     relationship_type: Optional[str] = Field(
#         default=None,
#         description="Type or nature of the relationship (e.g., 'friend', 'colleague', 'family')."
#     )

@weaviate_collection(name="ChunkPerson", vectors=["name", "context"], parent_id_field="chunk_id")
class Person(BaseModel):
    """
    Information that can be gleaned about one person from the diary entry.
    Leave any field empty / default if there is no clear evidence.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")

    name: Optional[str] = Field(
        default=None,
        description="The person's name or the clearest identifier that appears in the diary. "
                    "Leave empty if no unambiguous name/identifier is given."
    )
    alias: Optional[str] = Field(
        None,
        description="List of alias for this person."
    )
    description: Optional[str] = Field(
        None,
        description="Exact sentence(s) or a very short paraphrase of what the diary says about this person."
    )
    relationship_to_user: Optional[str] = Field(
        default=None,
        description="'positive', 'negative' or 'neutral' plus a supporting quote/paraphrase. "
                    "If the sentiment is inferred instead of explicitly stated, prefix with "
                    "'interpretation:'.  Leave empty if no information."
    )
    
    # relationships_to_others: Dict[str, str] = Field(
    #     default_factory=dict,
    #     description="Dictionary of explicit relationships this person has with other people "
    #                 "mentioned in the same diary entry. Keys are names/identifiers of other people, "
    #                 "values are short quotes or very brief paraphrases that evidence the relationship. "
    #                 "Only include relationships that are explicitly mentioned in the diary. "
    #                 "Leave empty dict if none."
    # )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class People(BaseModel):
    """
    Top-level container returned by the model.
    """
    items: List[Person]


# Thoughts & Reflections
@weaviate_collection(name="ChunkThought", vectors=["description", "title"], parent_id_field="chunk_id")
class ThoughtReflection(BaseModel):
    """
    One single thought or reflection extracted from a diary entry.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")

    description: str = Field(
        ...,
        description="What did the writer actually say?  Short paraphrase of statements + context."
    )
    title: str = Field(
        ...,
        description="Concise title for this thought/reflection."
    )
    emotion: str = Field(
        default="",
        description="Single-word emotion explicitly tied to the thought.  Empty string if none."
    )
    people_mentioned: List[str] = Field(
        ...,
        description="Person's names or nicknames exactly as they appear in the diary."
    )

    @field_validator('people_mentioned', mode='before')
    @classmethod
    def validate_people_mentioned(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v if v is not None else []

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class Thoughts(BaseModel):
    items: List[ThoughtReflection]


# emotion
@weaviate_collection(name="ChunkEmotion", vectors=["content", "title"], parent_id_field="chunk_id")
class Emotion(BaseModel):
    """
    One single feeling that can be inferred from the diary text.
    All fields are required – provide clear evidence from the diary entry for each piece of data.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")
    
    description: str = Field(
        ...,
        description="VERY short (≤12 words) quote-like description of what the user wrote in direct connection with the emotion."
    )
    title: str = Field(
        ...,
        max_length=50,
        description="One-word name of the emotion, capitalised (e.g. 'Happy', 'Anxious')."
    )
    intensity: str = Field(
        ...,
        description="Subjective intensity of the emotion as expressed by the user.",
        pattern="^(strong|normal|weak)$"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class Emotions(BaseModel):
    """
    Container for all emotions detected in a single diary chunk.
    The model must return a JSON object that is valid against this schema.
    """
    items: List[Emotion] = Field(
        default_factory=list,
        description="List of Emotion objects extracted from the diary chunk."
    )


# Problem
@weaviate_collection(name="ChunkProblem", vectors=["content", "title"], parent_id_field="chunk_id")
class Problem(BaseModel):
    """
    Exactly one object per element in the user-supplied `task_descriptions`
    list – unless two descriptions refer to the same unresolved issue, in
    which case they MAY be merged into one object.
    Leave string fields empty ("") and list fields empty ([]) if the diary
    entry does not contain clear evidence.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")
    description: str = Field(
        ...,
        description="Very short summary (one sentence, max ~15 words) of what "
                    "the diary author wrote in connection with this problem or concern."
    )
    title: str = Field(
        ...,
        max_length=50,
        description="One-word title that best captures the problem."
    )
    people: List[str] = Field(
        default_factory=list,
        description="Every person explicitly mentioned in the diary chunk in "
                    "connection with this problem. Use the literal names found "
                    "in the text."
    )
    emotions: List[str] = Field(
        default_factory=list,
        description="Emotions that are explicitly stated OR are plainly evident "
                    "from the context of this problem (e.g. 'anxious', 'guilty', "
                    "'sad').  Use single words, lower-case."
    )

    @field_validator('people', 'emotions', mode='before')
    @classmethod
    def validate_string_lists(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v if v is not None else []

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class Problems(BaseModel):
    """
    Root object returned by the model – contains the list of extracted problems.
    """
    items: List[Problem]


# achievement
@weaviate_collection(name="ChunkAchievement", vectors=["content", "title"], parent_id_field="chunk_id")
class Achievement(BaseModel):
    """
    One single achievement or accomplishment extracted from a diary entry.
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")
    description: str = Field(
        ...,
        description="What was achieved? Provide a short, factual statement."
    )
    title: str = Field(
        ...,
        max_length=50,
        description="One- or two-word summary title (capitalised)."
    )
    people: List[str] = Field(
        default_factory=list,
        description="People strongly connected to the achievement."
    )
    emotions: List[str] = Field(
        default_factory=list,
        description="Emotions explicitly mentioned or clearly implied."
    )

    @field_validator('people', 'emotions', mode='before')
    @classmethod
    def validate_string_lists(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v if v is not None else []

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())


class Achievements(BaseModel):
    items: List[Achievement]


# Future Intention
@weaviate_collection(name="ChunkFutureIntention", vectors=["content", "title"], parent_id_field="chunk_id")
class FutureIntention(BaseModel):
    """
    One single future-oriented item (plan, wish, scheduled event, promise, goal…)
    """
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Always return empty string - will be auto-generated")
    description: str = Field(
        ...,
        description="Description of the future event: what will / should happen?"
    )
    title: str = Field(
        ...,
        max_length=50,
        description="1–2-word title that sums the intention. No more than two words."
    )
    people: List[str] = Field(
        default_factory=list,
        description="List of people strongly connected to this intention. "
                    "Use an empty list if nobody is explicitly involved."
    )

    @field_validator('people', mode='before')
    @classmethod
    def validate_people(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v if v is not None else []

    def __init__(self, **data):
        super().__init__(**data)
        if not self.object_id:
            self.object_id = str(uuid.uuid4())

class Goals(BaseModel):
    """
    Container that groups all extracted future intentions for one diary chunk.
    """
    items: List[FutureIntention] = Field(
        ...,
        description="Exactly one item for every description in `descriptions`, "
                    "unless two descriptions clearly point to the same achievement, "
                    "in which case they must be merged into one."
    )


# outside llm

@weaviate_collection(name="Connection", vectors=[], parent_id_field="chunk_id")
class Connection(BaseModel):
    """Format for connections between entities"""
    source_id: str = Field(..., description="Object ID of the source object")
    target_id: str = Field(..., description="Object ID of the target object")
    type: str = Field(..., description="Specific type of connection in plain text (e.g., 'mentioned', 'participated in', 'related to', etc.)")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when connection was created")
    
class Connections(BaseModel):
    """Container for multiple connections"""
    items: List[Connection] = Field(
        default_factory=list,
        description="List of connections between entities"
    )

@weaviate_collection(name="Chunk", vectors=["summary"], subcollections={
    "events": "ChunkEvent",
    "people": "ChunkPerson",
    "thoughts": "ChunkThought",
    "emotions": "ChunkEmotion",
    "problems": "ChunkProblem",
    "achievements": "ChunkAchievement",
    "goals": "ChunkFutureIntention",
    "connections": "Connection"
})
class Chunk(BaseModel):
    """Format for chunk after Stage Two; Stage two Output!"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this chunk")
    original_text: str = Field(..., description="Original diary text for this chunk")
    summary: List[str] = Field(..., description="Summary of the chunk content")

    # stage two
    events: list[Event] | None  # python 3.1
    people: list[Person] | None
    thoughts: list[ThoughtReflection] | None
    emotions: list[Emotion] | None
    problems: list[Problem] | None
    achievements: list[Achievement] | None
    goals: list[FutureIntention] | None

    # connector
    connections: list[Connection] | None

    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when chunk was created")
    source_type: str = Field(default="diary", description="Type of source document")
    source_file: Optional[str] = Field(default=None, description="Source file path if applicable")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the chunk")
