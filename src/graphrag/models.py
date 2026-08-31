"""
GraphRAG data models.

Entity and relationship models that the extractor must conform to. The entity
``type`` maps 1:1 to a Neo4j node label (spaces removed); the relationship
``relation`` maps 1:1 to a Neo4j relationship type.

These models are the contract between the LLM extractor, the grounding guard
and the Neo4j writer — nothing outside this registry may be written to the
graph, which is what prevents fabricated/hallucinated nodes and edges.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Allowed entity categories (each becomes a Neo4j node label)."""

    MINISTRY = "Ministry"
    ORGANISATION = "Organisation"
    DEPARTMENT = "Department"
    INSTITUTE = "Institute"
    MISSION = "Mission"
    SCHEME = "Scheme"
    PROGRAMME = "Programme"
    STATE = "State"
    DISTRICT = "District"
    CITY = "City"
    RIVER = "River"
    OCEAN = "Ocean"
    SEA = "Sea"
    SATELLITE = "Satellite"
    RADAR = "Radar"
    WEATHER_SYSTEM = "WeatherSystem"
    CYCLONE = "Cyclone"
    TSUNAMI = "Tsunami"
    MONSOON = "Monsoon"
    CLIMATE_EVENT = "ClimateEvent"
    INSTRUMENT = "Instrument"
    SCIENTIST = "Scientist"
    COMMITTEE = "Committee"
    REPORT = "Report"
    PHENOMENON = "Phenomenon"
    TECHNOLOGY = "Technology"


class RelationType(str, Enum):
    """Allowed relationship types (each becomes a Neo4j relationship type)."""

    MENTIONS = "MENTIONS"
    LOCATED_IN = "LOCATED_IN"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"
    OPERATED_BY = "OPERATED_BY"
    FUNDED_BY = "FUNDED_BY"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    MONITORS = "MONITORS"
    FORECASTS = "FORECASTS"
    USES = "USES"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    REPORTS_TO = "REPORTS_TO"


# Map entity type -> Neo4j label (identity here, kept explicit for clarity)
ENTITY_LABELS: dict[str, str] = {t.value: t.value for t in EntityType}

ALLOWED_ENTITY_TYPES = {t.value for t in EntityType}
ALLOWED_RELATION_TYPES = {r.value for r in RelationType}

# Entity types that may serve as a relationship endpoint (all of them)
ENDPOINT_TYPES = ALLOWED_ENTITY_TYPES


class Entity(BaseModel):
    """A single extracted entity."""

    name: str = Field(..., min_length=2, max_length=200)
    type: EntityType
    # Optional source support: how the extractor grounded this entity.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = v.strip()
        # Collapse internal whitespace; reject pure-punctuation names.
        import re

        v = re.sub(r"\s+", " ", v)
        if not re.search(r"[A-Za-z0-9]", v):
            raise ValueError(f"entity name has no alphanumeric content: {v!r}")
        return v[:200]

    @property
    def label(self) -> str:
        """Neo4j node label for this entity."""
        return self.type.value


class Relationship(BaseModel):
    """A single extracted relationship between two entities."""

    source_type: EntityType
    source_name: str
    relation: RelationType
    target_type: EntityType
    target_name: str
    # Short verbatim snippet from the parliamentary answer that supports the
    # relationship (used for grounding / auditability).
    evidence: Optional[str] = Field(default=None, max_length=600)


class DocumentRecord(BaseModel):
    """A Q&A document record as read from the enriched JSONL."""

    question_id: str
    question_text: str
    answer_text: str
    ministry: Optional[str] = None
    subject: Optional[str] = None
    session: Optional[int] = None
    question_number: Optional[int] = None
    parliament_number: Optional[int] = None
    date: Optional[str] = None
    source_url: Optional[str] = None
