"""
Data models for Parliamentary Q&A records.

Design Decisions
----------------
1. We model the Q&A as a single document unit — not chunked.
   Rationale: Each Q&A pair is self-contained; the answer is meaningless
   without its question. Splitting them would break retrieval integrity.

2. We separate core content (question_text, answer_text) from metadata.
   Rationale: The core content is what's retrieved and passed to the LLM.
   Metadata enriches the record for filtering and analysis but is not
   part of the retrieval corpus.

3. Question IDs follow the Lok Sabha numbering convention: "{SessionNo}-{QuestionNo}"
   Example: "18-101" means Session 18, Question 101.
   This is the most reliable deduplication key.

4. answer_text can be a URL reference for very long answers.
   Rationale: Lok Sabha answers can be extremely long (10,000+ words).
   We store the full text but note when it's a reference to avoid confusion.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class QuestionType(str, Enum):
    """Type of parliamentary question, following Lok Sabha conventions."""

    STARRED = "starred"           # MP's name recorded; answered orally
    UNSTARRED = "unstarred"       # Written answer; MP's name not recorded
    SHORT_NOTICE = "short_notice" # Asked with less notice; urgent matters
    HALF_HOUR = "half_hour"       # Discussion without formal question
    PRIVILEGE = "privilege"       # Matters of parliamentary privilege
    ADJOURNMENT = "adjournment"   # Motion to adjourn
    CALLING_ATTENTION = "calling_attention"  # Minister draws attention to matter
    UNKNOWN = "unknown"


class AnswerStatus(str, Enum):
    """Whether the answer was actually provided."""

    ANSWERED = "answered"
    PARTIAL = "partial"        # Partially answered
    UNANSWERED = "unanswered"  # Due but not provided
    REFERRED = "referred"       # Referred to another ministry
    WITHDRAWN = "withdrawn"     # Withdrawn by MP
    NOT_PRESENT = "not_present" # MP not present


class QARecordMetadata(BaseModel):
    """
    Non-content metadata extracted from the Q&A record.
    These fields may be absent when scraped from raw pages.
    """

    ministry: Optional[str] = Field(
        default=None,
        description="Ministry/department to which the question was addressed",
        examples=["Finance", "Health and Family Welfare"],
    )
    date: Optional[str] = Field(
        default=None,
        description="Date the question was asked (or answered session date)",
        examples=["2023-03-14", "14 March 2023"],
    )
    session: Optional[int] = Field(
        default=None,
        description="Parliamentary session number",
        ge=1,
        examples=[17, 18],
    )
    sitting: Optional[int] = Field(
        default=None,
        description="Sitting number within the session",
        ge=1,
    )
    question_number: Optional[int] = Field(
        default=None,
        description="Question number within the session",
        ge=1,
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject/topic classification of the question",
        examples=["Malaria Control", "Rural Electrification"],
    )
    question_type: QuestionType = Field(
        default=QuestionType.UNKNOWN,
        description="Type of parliamentary question",
    )
    answer_status: AnswerStatus = Field(
        default=AnswerStatus.ANSWERED,
        description="Whether the answer was provided",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original source URL on sansad.in",
        examples=["https://sansad.in/ls/questions/questions-and-answers/..."],
    )
    parliament_number: Optional[int] = Field(
        default=None,
        description="Lok Sabha parliament number (e.g., 17th Lok Sabha)",
        ge=1,
        le=18,
    )


class QARecord(BaseModel):
    """
    A single Lok Sabha Question & Answer record.

    This is the atomic unit of our knowledge base.
    We treat one Q&A pair as one retrievable document — no chunking.

    Attributes
    ----------
    question_id : str
        Unique identifier. Format: "{SessionNo}-{QuestionNo}" or
        a hash-based fallback if that can't be determined.
    question_text : str
        The full question as asked by the MP.
    answer_text : str
        The full official answer from the ministry.
        May be a URL reference if the answer is very long.
    metadata : QARecordMetadata
        All non-content fields.
    scraped_at : datetime
        When this record was scraped (UTC).

    Validation Rules
    ----------------
    - question_text and answer_text are both required and must be non-empty.
    - question_text must be at least 10 characters (minimum meaningful content).
    - answer_text must be at least 10 characters.
    - question_id is auto-generated from metadata if not provided.
    """

    question_id: str = Field(
        ...,
        description="Unique identifier for this Q&A record",
        min_length=1,
    )
    question_text: str = Field(
        ...,
        description="The full question text as asked by the MP",
        min_length=10,
    )
    answer_text: str = Field(
        ...,
        description="The full official answer text",
        min_length=10,
    )
    metadata: QARecordMetadata = Field(
        default_factory=QARecordMetadata,
        description="Metadata associated with this Q&A",
    )
    scraped_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this record was scraped",
    )

    @field_validator("question_text", "answer_text", mode="before")
    @classmethod
    def strip_and_validate_not_empty(cls, v: Any) -> str:
        """Strip whitespace; reject empty strings."""
        if not isinstance(v, str):
            raise TypeError(f"Expected string, got {type(v).__name__}")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only")
        return stripped

    @field_validator("question_id", mode="before")
    @classmethod
    def normalize_question_id(cls, v: Any) -> str:
        """Normalize question ID to a clean string."""
        if not isinstance(v, str):
            raise TypeError(f"question_id must be string, got {type(v).__name__}")
        # Remove whitespace, normalize dashes
        normalized = re.sub(r"[\s—–-]+", "-", v.strip())
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized or "unknown"

    @computed_field
    @property
    def document_content(self) -> str:
        """
        The combined text used for embedding and BM25 indexing.

        Format:
        --------
        QUESTION: {question_text}

        ANSWER: {answer_text}

        Metadata: [metadata line if available]

        Rationale for this format:
        - BM25 benefits from clear section markers
        - LLM benefits from the Q→A structure being explicit
        - Metadata is available but not dominant in the embedding space
        """
        parts = [
            f"QUESTION: {self.question_text}",
            f"ANSWER: {self.answer_text}",
        ]
        if self.metadata.ministry:
            parts.append(f"MINISTRY: {self.metadata.ministry}")
        if self.metadata.subject:
            parts.append(f"SUBJECT: {self.metadata.subject}")
        if self.metadata.question_type and self.metadata.question_type != QuestionType.UNKNOWN:
            parts.append(f"QUESTION_TYPE: {self.metadata.question_type.value}")
        return "\n\n".join(parts)

    @computed_field
    @property
    def content_hash(self) -> str:
        """Stable hash of question_text for deduplication."""
        return hashlib.sha256(
            self.question_text.encode("utf-8")
        ).hexdigest()[:16]

    @model_validator(mode="after")
    def auto_generate_question_id(self) -> "QARecord":
        """
        Auto-generate question_id if not meaningfully set.

        If the question_id is just 'unknown' or empty, generate one from
        metadata fields or the content hash.
        """
        if self.question_id in ("", "unknown", "Unknown"):
            # Try to build from metadata
            parts = []
            if self.metadata.session:
                parts.append(str(self.metadata.session))
            if self.metadata.question_number:
                parts.append(str(self.metadata.question_number))
            if parts:
                self.question_id = "-".join(parts)
            else:
                # Fall back to content hash
                self.question_id = f"gen-{self.content_hash}"
        return self

    def to_document_dict(self) -> dict[str, Any]:
        """
        Convert to a dict suitable for indexing in retrieval systems.
        Returns only the fields needed for retrieval — not full metadata.
        """
        return {
            "doc_id": self.question_id,
            "content": self.document_content,
            "question": self.question_text,
            "answer": self.answer_text,
            "metadata": {
                "ministry": self.metadata.ministry,
                "subject": self.metadata.subject,
                "question_type": self.metadata.question_type.value,
                "date": self.metadata.date,
                "session": self.metadata.session,
            },
        }

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
    }
