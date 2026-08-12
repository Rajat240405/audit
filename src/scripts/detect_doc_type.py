"""
Smart document-type detection for ingested files.

Order of signals (first match wins):
  1. FOLDER name hint   — the folder the file lives in
     (annual_reports, ResearchPublications, TechnicalReports, Others, ...)
  2. FILENAME pattern   — AR_* / TR_* / RP_* / Report_* / MoES
  3. CONTENT header     — first page/first ~800 chars: "ANNUAL REPORT",
     "TECHNICAL REPORT", "RESEARCH PUBLICATION", etc.

Returns one of: annual_report | technical_report | research_publication |
general_report | audit_qa | document (default).
"""

from __future__ import annotations

from pathlib import Path


def detect_doc_type(path: Path, text: str = "", folder_hint: str | None = None) -> str:
    name = path.name.lower()
    folder = (folder_hint or path.parent.name).lower()

    # ── 1. Content header — the document literally declaring itself is the
    #    most reliable signal (beats filename AND ambiguous folder) ──
    head = (text or "")[:800].lower()
    if "annual report" in head:
        return "annual_report"
    if "technical report" in head:
        return "technical_report"
    if "research publication" in head or "journal of" in head:
        return "research_publication"

    # ── 2. Folder hint (strong — scientist put it in this folder) ──
    if "annual" in folder:
        # sir's convention: Report_* inside AnnualReports are GENERAL reports
        if name.startswith("report_"):
            return "general_report"
        return "annual_report"
    if "researchpublication" in folder.replace("_", "").replace(" ", "") or "research" in folder:
        return "research_publication"
    if "technical" in folder or "techreport" in folder.replace(" ", ""):
        return "technical_report"
    if folder in ("others", "general", "generalreport"):
        return "general_report"
    if "moes" in folder:
        return "document"

    # ── 3. Filename pattern (reliable for AR_/TR_/RP_) ──
    if name.startswith("ar_"):
        return "annual_report"
    if name.startswith("tr_") or "technicalreport" in name.replace("_", ""):
        return "technical_report"
    if name.startswith("rp_"):
        return "research_publication"
    if name.startswith("report_"):
        return "general_report"

    # ── 4. Weak content fallback ──
    if head.startswith(("report", "project report")):
        return "general_report"

    return "document"


def readable_type(doc_type: str) -> str:
    """Human label for a detected document type."""
    return {
        "annual_report": "INCOIS Annual Report",
        "technical_report": "INCOIS Technical Report",
        "research_publication": "Research Publication",
        "general_report": "INCOIS General Report",
        "audit_qa": "Audit Q&A",
        "document": "Document",
    }.get(doc_type, doc_type)
