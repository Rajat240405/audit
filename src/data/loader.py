"""
Data loader utilities for reading/writing Q&A records.

Handles multiple formats (JSON, JSONL) and provides
a unified interface for loading records for downstream pipelines.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import orjson

from src.models.qa_record import QARecord


class DataLoader:
    """
    Loads and saves Q&A records in various formats.

    Supports:
    - JSONL (one record per line) — preferred for large datasets
    - JSON (array of records) — for small datasets or human readability

    Design Decision: JSONL is the preferred format because:
    1. Streaming reads — don't need to load entire file into memory
    2. Append-friendly — can append new records without rewriting
    3. Line-based — easy to inspect with `head`, `grep`, `wc -l`
    4. Standard for data pipelines (Bloomberg, Airbnb, etc.)
    """

    @staticmethod
    def load_jsonl(
        path: str | Path,
        limit: int | None = None,
    ) -> list[QARecord]:
        """
        Load records from a JSONL file.

        Parameters
        ----------
        path : str | Path
            Path to the .jsonl file.
        limit : int, optional
            Maximum number of records to load. Loads all if None.

        Returns
        -------
        list[QARecord]
        """
        records: list[QARecord] = []
        count = 0
        with open(path, "rb") as f:
            for line in f:
                if limit and count >= limit:
                    break
                if not line.strip():
                    continue
                data = orjson.loads(line)
                records.append(QARecord.model_validate(data))
                count += 1
        return records

    @staticmethod
    def load_jsonl_streaming(
        path: str | Path,
    ) -> Iterator[QARecord]:
        """
        Load records from a JSONL file as a generator (memory-efficient).

        Use this for very large files where loading all records
        into memory is not feasible.
        """
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                data = orjson.loads(line)
                yield QARecord.model_validate(data)

    @staticmethod
    def load_json(
        path: str | Path,
    ) -> list[QARecord]:
        """Load records from a JSON file (array format)."""
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
        return [QARecord.model_validate(item) for item in data]

    @staticmethod
    def save_jsonl(
        records: list[QARecord],
        path: str | Path,
        append: bool = False,
    ) -> None:
        """
        Save records to a JSONL file.

        Parameters
        ----------
        records : list[QARecord]
            Records to save.
        path : str | Path
            Output path.
        append : bool
            If True, append to existing file. If False, overwrite.
        """
        path = Path(path)
        mode = "ab" if append else "wb"
        with open(path, mode) as f:
            for record in records:
                f.write(orjson.dumps(record.model_dump(mode="json")) + b"\n")

    @staticmethod
    def save_json(
        records: list[QARecord],
        path: str | Path,
        indent: bool = True,
    ) -> None:
        """Save records to a JSON file (array format)."""
        path = Path(path)
        data = [record.model_dump(mode="json") for record in records]
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else 0))

    @staticmethod
    def count_records(path: str | Path) -> int:
        """Count the number of records in a JSONL file without loading them."""
        count = 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
