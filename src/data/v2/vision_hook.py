"""Stage 11 — optional generation-time vision hook. **Disabled by default.**

The spec (§13.7) records that no vision model is provisioned and no dependency
exists for one. So this module is a *hook*, not a feature: a single narrow place
where a future model could describe a figure crop, wired so that its absence is
indistinguishable from its being switched off.

Contract:

* ``V2_VISION_AT_GENERATION`` defaults to **false**;
* no vision library is imported at module scope, or ever unless enabled;
* every failure path — disabled, no model, unreadable crop, model error —
  returns the **textual figure-card evidence unchanged**. There is no code path
  in which enabling vision can make generation worse than not having it, and
  none in which disabling it loses evidence;
* the normal production generation client never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.v2 import evidence
from src.data.v2.config import V2Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: Cached probe result: (available, reason).
_probe: tuple[bool, str] | None = None

#: Module names that would provide a vision model. Checked, never required.
VISION_BACKENDS = ("transformers", "torch")


def reset_probe() -> None:
    """Clear the cached availability probe (tests, post-install checks)."""
    global _probe  # noqa: PLW0603
    _probe = None


def _importable(name: str) -> bool:
    """Can ``name`` be imported? Never raises.

    ``importlib.util.find_spec`` raises ``ValueError`` when a module is already
    in ``sys.modules`` with ``__spec__ = None`` — which is exactly what a test
    stub looks like. A stubbed backend is not an installed one, so it must
    report unavailable rather than crash the probe.
    """
    import importlib.util
    import sys

    if name in sys.modules:
        module = sys.modules[name]
        return getattr(module, "__spec__", None) is not None and getattr(
            module, "__file__", None
        ) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def vision_available() -> tuple[bool, str]:
    """Is a vision model actually usable here? Cached, never raises."""
    global _probe  # noqa: PLW0603
    if _probe is not None:
        return _probe

    missing = [name for name in VISION_BACKENDS if not _importable(name)]
    if missing:
        _probe = (False, f"vision backend(s) not installed: {','.join(missing)}")
        return _probe
    _probe = (True, "backend importable (no model verified)")
    return _probe


def describe_crop(crop_path: str | Path, config: V2Config | None = None) -> str | None:
    """Describe a figure crop, or return ``None``.

    ``None`` means "no description available" for *any* reason — disabled, no
    backend, missing file, model error. Callers must treat it as absent, not as
    an empty description.
    """
    cfg = config if config is not None else V2Config()
    if not cfg.vision_active:
        return None

    available, _reason = vision_available()
    if not available:
        return None

    path = Path(crop_path)
    if not path.is_file():
        return None

    # A real implementation would load a multimodal model here and return its
    # caption. Deliberately NOT implemented: spec §13.7 records no provisioned
    # model, and shipping a call to an unpinned model would make generation
    # nondeterministic and add a multi-gigabyte dependency to a path that
    # currently needs none. Returning None keeps the textual fallback in force.
    return None


def figure_evidence(
    card: dict | Any, config: V2Config | None = None
) -> tuple[str, dict[str, Any]]:
    """Evidence text for a figure card, plus how it was produced.

    Returns ``(text, provenance)``. ``provenance["source"]`` is ``"vision"`` or
    ``"textual"`` so a caller (or an audit) can tell which path produced the
    evidence. With vision off — the default — this is exactly
    :func:`evidence.render_figure_evidence`.
    """
    cfg = config if config is not None else V2Config()
    textual = evidence.render_figure_evidence(card)

    data = card if isinstance(card, dict) else evidence._as_dict(card)  # noqa: SLF001
    crop = data.get("crop")
    if not crop or not cfg.vision_active:
        return textual, {"source": "textual", "vision_attempted": False}

    description = describe_crop(crop, cfg)
    if not description:
        return textual, {
            "source": "textual",
            "vision_attempted": True,
            "vision_reason": vision_available()[1],
        }

    return (
        textual + f"\nvision: {description}",
        {"source": "vision", "vision_attempted": True},
    )


__all__ = [
    "VISION_BACKENDS",
    "describe_crop",
    "figure_evidence",
    "reset_probe",
    "vision_available",
]
