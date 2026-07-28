"""Load and format ChartProbe question templates.

Default file: ``src/question_templates.json``.

Customize probes by editing that JSON (or pass a custom path)::

    from templates import load_templates, get_template, format_question

    specs = load_templates()                       # default JSON
    specs = load_templates(Path("my_templates.json"))
    tmpl = get_template(specs, "bar", "P-HEIGHT-2X")
    q = format_question(tmpl, ...)                # fill {placeholders}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_PATH = SCRIPT_DIR / "question_templates.json"

_CACHE: dict[str, dict[str, Any]] = {}


def load_templates(path: Path | str | None = None) -> dict[str, Any]:
    """Load a templates JSON. Results are cached by resolved path."""
    p = Path(path) if path is not None else DEFAULT_TEMPLATES_PATH
    p = p.expanduser().resolve()
    key = str(p)
    if key not in _CACHE:
        if not p.is_file():
            raise FileNotFoundError(f"Question templates not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if "families" not in data:
            raise ValueError(f"Invalid templates file (missing 'families'): {p}")
        _CACHE[key] = data
    return _CACHE[key]


def clear_templates_cache() -> None:
    _CACHE.clear()


def list_families(specs: dict[str, Any] | None = None) -> list[str]:
    specs = specs or load_templates()
    return sorted(specs["families"].keys())


def family_spec(specs: dict[str, Any], family: str) -> dict[str, Any]:
    families = specs["families"]
    if family not in families:
        raise KeyError(f"Unknown template family {family!r}. Known: {sorted(families)}")
    return families[family]


def iter_templates(specs: dict[str, Any], family: str) -> list[dict[str, Any]]:
    return list(family_spec(specs, family).get("templates") or [])


def templates_by_id(specs: dict[str, Any], family: str) -> dict[str, dict[str, Any]]:
    return {t["template_id"]: t for t in iter_templates(specs, family)}


def get_template(specs: dict[str, Any], family: str, template_id: str) -> dict[str, Any]:
    by_id = templates_by_id(specs, family)
    if template_id not in by_id:
        # Allow P-COUNT-CUBE when only P-COUNT-{SHAPE} is listed, etc.
        if template_id.startswith("P-COUNT-") and "P-COUNT-{SHAPE}" in by_id:
            shape = template_id.removeprefix("P-COUNT-").lower()
            base = dict(by_id["P-COUNT-{SHAPE}"])
            base["template_id"] = template_id
            base["question_template"] = format_question(base, shape=shape)
            return base
        raise KeyError(
            f"Unknown template_id {template_id!r} in family {family!r}. "
            f"Known: {sorted(by_id)}"
        )
    return by_id[template_id]


def format_question(template: dict[str, Any] | str, **slots: Any) -> str:
    """Fill ``{placeholders}`` in ``question_template`` (or a raw string)."""
    if isinstance(template, dict):
        text = template["question_template"]
    else:
        text = template
    try:
        return text.format(**slots)
    except KeyError as e:
        raise KeyError(
            f"Missing template slot {e.args[0]!r} for question {text!r}. "
            f"Provided: {sorted(slots)}"
        ) from e


def clevr_test_set_selection(specs: dict[str, Any] | None = None) -> dict[str, Any]:
    specs = specs or load_templates()
    sel = family_spec(specs, "clevr").get("test_set_selection")
    if not sel:
        raise KeyError("clevr family is missing test_set_selection")
    return sel
