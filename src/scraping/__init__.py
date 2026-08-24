"""Crawler framework (scraping architecture plan).

Houses the shared machinery used by the house-specific crawlers
(``src/scraping/rs`` today; ``src/scraping/ls`` later). Crawlers STAGE data
only — they download official records/documents into the corpus hierarchy
(``data/parliamentary-qa/<house>/session-<n>/``) and never touch
embedding/indexing/ingestion (that stays in the existing engine).
"""
