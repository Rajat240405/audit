"""MoES website crawler (staging-only, additive).

Crawls the ministry-owned CMS post/document layer of www.moes.gov.in into
``data/.moes-website/`` (staging root — hidden on purpose: neither claimed by
the registered hierarchical ``moes`` ingestion source nor auto-discovered by
``ingest``). v1 approved scope: reports families annual-reports /
monthly-reports / demands-for-grants, and the whole press-release category.
Everything else in this package is scope-guarded fail-closed.

Reuses the frozen shared framework (``src/scraping/{http,formats,records,
manifest}.py``) without modifying it; MoES-specific behaviour lives here.
"""
