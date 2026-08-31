"""Lok Sabha crawler (sansad.in api_ls + elibrary.sansad.in DSpace sources).

Implements the validated design document
``investigation_scraping_architecture/ls_scraper_plan/REPORT.md``.
Two discovery eras, one staging contract:

- **modern era** (Lok Sabha ≥ 16, config ``eras.api_ls_min_loksabha``):
  inventory from ``sansad.in/api_ls/question/qetFilteredQuestionsAns``
  (Referer + browser-UA required — anonymous 403 otherwise).
- **legacy era** (Lok Sabha ≤ 15): inventory from the elibrary DSpace 7
  anonymous discover search ``/server/api/discover/search/objects`` with
  ``f.ministry`` / ``f.loksabhanumber`` facets; documents resolve through
  ``src/scraping/dspace.resolve_handle`` (REST ladder → HTML fallback).

Lok Sabha ONLY — the frozen workbook
(``data/raw/Loksabha_questions.xlsx``) is reference/fixture data, never the
scraper input. Crawlers STAGE data only; ingestion stays in
``src/scripts/ingest.py`` (``lok_sabha`` records source).
"""
