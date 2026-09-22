"""Shared curated-knowledge subsystem.

``store``    one file per contribution, UUID-keyed, atomic writes
``match``    exact -> semantic -> ambiguity-gate cascade
``migrate``  explicit, idempotent, non-destructive upgrade of legacy files
"""
