"""
Optional metadata enrichment for Q&A records.

Design Decisions
----------------
1. Enrichment is strictly optional — it enhances records but the system
   works without it. This aligns with the architecture principle that
   metadata is an enhancement, not a dependency.

2. We use regex-based rules for known patterns (ministry names, dates,
   question numbers) because they're fast and don't require an LLM call.

3. Ministry extraction uses a curated list of known Lok Sabha ministries.
   Any unmatched text is left as-is — we don't hallucinate ministry names.

4. Date extraction uses multiple format patterns to handle the variety
   of date formats in parliamentary records.

5. Question number extraction uses the Lok Sabha URL convention and
   the standard "Q. No." prefix patterns.

6. Subject extraction attempts to identify the topic from the question
   text — this is intentionally approximate.
"""

from __future__ import annotations

import re

from src.models.qa_record import QARecord, QuestionType

# Curated list of Lok Sabha ministries — used for validation and extraction
KNOWN_MINISTRIES = [
    "Finance",
    "Health and Family Welfare",
    "Health and Family",
    "Education",
    "Home Affairs",
    "External Affairs",
    "Defence",
    "Agriculture and Farmers Welfare",
    "Agriculture",
    "Railways",
    "Road Transport and Highways",
    "Power",
    "Drinking Water and Sanitation",
    "Housing and Urban Affairs",
    "Labour and Employment",
    "Women and Child Development",
    "Environment, Forest and Climate Change",
    "Textiles",
    "Commerce and Industry",
    "Petroleum and Natural Gas",
    "Tourism",
    "Culture",
    "Information and Broadcasting",
    "Consumer Affairs, Food and Public Distribution",
    "Consumer Affairs",
    "Food and Public Distribution",
    "Law and Justice",
    "Electronics and Information Technology",
    "Ayush",
    "Ports, Shipping and Waterways",
    "Skill Development and Entrepreneurship",
    "Micro, Small and Medium Enterprises",
    "Housing and Urban Affairs",
    "Science and Technology",
    "Earth Sciences",
    "Jal Shakti",
    "Water Resources",
    "Cooperation",
    "Panchayati Raj",
    "Tribal Affairs",
    "Social Justice and Empowerment",
    "Minority Affairs",
    "Rural Development",
    "Urban Development",
    "Labour",
]

# Build a case-insensitive lookup dict
MINISTRY_LOOKUP = {m.lower(): m for m in KNOWN_MINISTRIES}

# Patterns for ministry detection
MINISTRY_PATTERNS = [
    re.compile(
        r"(?:to|for|addressed to|regarding|on)\s+[:.]?\s*"
        r"(the\s+)?"  # optional 'the'
        r"("
        + "|".join(re.escape(m.lower()) for m in KNOWN_MINISTRIES)
        + r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Minister|MoS|Secretary)\s+(?:of|for)\s+(?:the\s+)?"
        r"("
        + "|".join(re.escape(m.lower()) for m in KNOWN_MINISTRIES)
        + r")",
        re.IGNORECASE,
    ),
]

# Date patterns in parliamentary records
DATE_PATTERNS = [
    # 14 March 2023, 14/03/2023, 2023-03-14, March 14, 2023
    (re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})\b", re.IGNORECASE), "day_month_year"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "year_month_day"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "day_month_year_slash"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE), "month_day_year"),
]

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Question number patterns
QUESTION_NUM_PATTERNS = [
    re.compile(r"#(\d+)", re.IGNORECASE),
    re.compile(r"(?:Q\.?No?\.?\s*:?\s*)(\d+)", re.IGNORECASE),
    re.compile(r"(?:Qstn\s*No?\.?\s*:?\s*)(\d+)", re.IGNORECASE),
    re.compile(r"(?:Serial\s*No?\.?\s*:?\s*)(\d+)", re.IGNORECASE),
]

# Session patterns
SESSION_PATTERNS = [
    re.compile(r"(?:Session|Lok Sabha)\s*(?:No\.?\s*)?(\d+)", re.IGNORECASE),
    re.compile(r"(?:(?:18th|17th|16th|15th)\s+Lok\s+Sabha)", re.IGNORECASE),
]

# Question type patterns
QUESTION_TYPE_PATTERNS = [
    (re.compile(r"\bSTARRED\b", re.IGNORECASE), QuestionType.STARRED),
    (re.compile(r"\bUNSTARRED\b", re.IGNORECASE), QuestionType.UNSTARRED),
    (re.compile(r"\bSHORT NOTICE\b", re.IGNORECASE), QuestionType.SHORT_NOTICE),
    (re.compile(r"\bHALF.HOUR\b", re.IGNORECASE), QuestionType.HALF_HOUR),
    (re.compile(r"\bPRIVILEGE\b", re.IGNORECASE), QuestionType.PRIVILEGE),
]

# Subject keywords for topic extraction
SUBJECT_KEYWORDS = {
    "health": ["health", "malaria", "dengue", "covid", "hospital", "doctor", "vaccine", "medical", "disease", "sanitation"],
    "education": ["education", "school", "university", "student", "teacher", "literacy", "skill", "training"],
    "finance": ["budget", "tax", "gst", "bank", "loan", "credit", "economy", "fiscal", "revenue", "expenditure"],
    "agriculture": ["farm", "agriculture", "crop", "farmer", "mandi", "msp", "fertilizer", "seeds", "irrigation"],
    "defence": ["defence", "military", "army", "navy", "air force", "border", "security", "terrorist"],
    "infrastructure": ["road", "highway", "bridge", "railway", "metro", "airport", "port", "construction"],
    "water": ["water", "river", "dam", "irrigation", "flood", "groundwater", "drinking water"],
    "environment": ["environment", "pollution", "forest", "climate", "wildlife", "carbon", "emission"],
    "women": ["women", "child", "nutrition", "anganwadi", "maternal", "gender"],
    "rural": ["rural", "village", "panchayat", "mgnrega", "swachh bharat", "toilet"],
    "energy": ["power", "electricity", "solar", "wind", "renewable", "coal", "energy", "oil", "gas", "petroleum"],
    "employment": ["job", "employment", "unemployment", "labour", "worker", "msme", "startup"],
    "digital": ["digital", "internet", "broadband", "mobile", "cybersecurity", "it", "software", "ai"],
}


class DataEnricher:
    """
    Enriches Q&A records with metadata extracted from text.

    All enrichment is optional and non-destructive — existing metadata
    is preserved unless a better value can be extracted.

    Usage
    -----
    ```python
    enricher = DataEnricher()
    enriched = enricher.enrich(qa_record)
    ```

    Or in batch:

    ```python
    enricher = DataEnricher()
    for record in records:
        enricher.enrich_inplace(record)
    ```
    """

    def __init__(self, strict: bool = False) -> None:
        """
        Parameters
        ----------
        strict : bool
            If True, override existing metadata with extracted values.
            If False, only fill in None values (preserve what was scraped).
        """
        self.strict = strict

    def enrich(self, record: QARecord) -> QARecord:
        """
        Create an enriched copy of the record.
        Does not modify the original.
        """
        enriched = QARecord.model_validate(record.model_dump())
        self.enrich_inplace(enriched)
        return enriched

    def enrich_inplace(self, record: QARecord) -> None:
        """
        Enrich a record in-place.

        Applies enrichment in this order:
        1. Ministry extraction
        2. Date extraction
        3. Question number extraction
        4. Session extraction
        5. Question type detection
        6. Subject tagging
        """
        combined_text = f"{record.question_text} {record.answer_text}".lower()

        # 1. Ministry
        if self._should_fill(record.metadata.ministry):
            record.metadata.ministry = self._extract_ministry(record.question_text)

        # 2. Date
        if self._should_fill(record.metadata.date):
            record.metadata.date = self._extract_date(combined_text)

        # 3. Question number
        if self._should_fill(record.metadata.question_number):
            record.metadata.question_number = self._extract_question_number(
                record.question_text
            ) or record.metadata.question_number

        # 4. Session
        if self._should_fill(record.metadata.session):
            record.metadata.session = self._extract_session(combined_text)

        # 5. Question type
        if record.metadata.question_type in (QuestionType.UNKNOWN, None):
            record.metadata.question_type = self._detect_question_type(combined_text)

        # 6. Subject
        if self._should_fill(record.metadata.subject):
            record.metadata.subject = self._extract_subject(combined_text)

    def _should_fill(self, current_value: str | int | None) -> bool:
        """Return True if we should try to fill this field."""
        if self.strict:
            return True
        if current_value is None:
            return True
        # Handle both str and int fields
        str_val = str(current_value).strip() if current_value else ""
        return str_val == ""

    def _extract_ministry(self, text: str) -> str | None:
        """Extract ministry name from question text."""
        text_lower = text.lower()
        for pattern in MINISTRY_PATTERNS:
            match = pattern.search(text)
            if match:
                # Try each capture group (some patterns have 2 groups, some have 1)
                found = None
                for group_idx in range(1, match.lastindex + 2):
                    try:
                        candidate = match.group(group_idx)
                        if candidate:
                            found = candidate
                            break
                    except (IndexError, AttributeError):
                        continue

                if not found:
                    continue

                found_lower = found.strip().lower()
                # Direct lookup
                if found_lower in MINISTRY_LOOKUP:
                    return MINISTRY_LOOKUP[found_lower]
                # Partial match
                for k, v in MINISTRY_LOOKUP.items():
                    if k in found_lower or found_lower in k:
                        return v
        return None

    def _extract_date(self, text: str) -> str | None:
        """Extract date from text in ISO format."""
        for pattern, fmt in DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                try:
                    if fmt == "day_month_year":
                        day, month_abbr, year = groups
                        month = MONTH_MAP.get(month_abbr[:3].lower(), "01")
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    elif fmt == "year_month_day":
                        year, month, day = groups
                        return f"{year}-{month}-{day}"
                    elif fmt == "day_month_year_slash":
                        d, m, y = groups
                        return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    elif fmt == "month_day_year":
                        month_abbr, day, year = groups
                        month = MONTH_MAP.get(month_abbr[:3].lower(), "01")
                        return f"{year}-{month}-{day.zfill(2)}"
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_question_number(self, text: str) -> int | None:
        """Extract question number from question text."""
        for pattern in QUESTION_NUM_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_session(self, text: str) -> int | None:
        """Extract parliamentary session number from text."""
        for pattern in SESSION_PATTERNS:
            match = pattern.search(text)
            if match:
                if match.lastindex and match.group(1):
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
                # Check for named Lok Sabha references
                sabha_match = re.search(r"(18th|17th|16th|15th|14th)\s+Lok\s+Sabha", text)
                if sabha_match:
                    numeral = sabha_match.group(1)
                    return {"18th": 18, "17th": 17, "16th": 16, "15th": 15, "14th": 14}.get(numeral)
        return None

    def _detect_question_type(self, text: str) -> QuestionType:
        """Detect the type of parliamentary question."""
        for pattern, qtype in QUESTION_TYPE_PATTERNS:
            if pattern.search(text):
                return qtype
        return QuestionType.UNKNOWN

    def _extract_subject(self, text: str) -> str | None:
        """Extract a high-level subject/topic from question text."""
        scores: dict[str, int] = {}
        for subject, keywords in SUBJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[subject] = score

        if not scores:
            return None

        # Return the highest-scoring subject
        best = max(scores, key=lambda s: scores[s])
        # Return in title case
        return best.title().replace("_", " ")

    def enrich_batch(
        self,
        records: list[QARecord],
        show_progress: bool = True,
    ) -> list[QARecord]:
        """Enrich a batch of records."""
        from rich.progress import Progress, TextColumn

        if show_progress:
            from rich.console import Console
            console = Console()
            with Progress(TextColumn("[progress.description]{task.description}"), console=console) as p:
                task = p.add_task("Enriching records...", total=len(records))
                for record in records:
                    self.enrich_inplace(record)
                    p.advance(task)
            return records
        else:
            for record in records:
                self.enrich_inplace(record)
            return records
