from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ValidationError

_REQUIRED_RANGES = ("sequence_length", "image_height", "image_width", "image_count", "state_length", "action_horizon")


def load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read shape profile {source}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValidationError("non-JSON YAML profiles require: pip install aginfer[yaml]") from exc
        try:
            value = yaml.safe_load(text)
        except Exception as exc:
            raise ValidationError(f"invalid shape profile {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("shape profile must be an object")
    profiles = value.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValidationError("shape profile requires a non-empty profiles list")
    names: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ValidationError(f"profile {index} must be an object")
        name = profile.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValidationError(f"profile {index} needs a unique non-empty name")
        names.add(name)
        if profile.get("batch", 1) != 1:
            raise ValidationError(f"profile {name}: only batch=1 is supported")
        for field in _REQUIRED_RANGES:
            bounds = profile.get(field)
            if not isinstance(bounds, dict) or set(bounds) != {"min", "opt", "max"}:
                raise ValidationError(f"profile {name}: {field} must define min, opt, and max")
            try:
                minimum, optimum, maximum = (int(bounds[key]) for key in ("min", "opt", "max"))
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"profile {name}: {field} bounds must be integers") from exc
            if minimum <= 0 or not minimum <= optimum <= maximum:
                raise ValidationError(f"profile {name}: expected 0 < min <= opt <= max for {field}")
    denoising = value.get("default_denoising", {})
    if not isinstance(denoising, dict) or int(denoising.get("steps", 1)) <= 0:
        raise ValidationError("default_denoising.steps must be positive")
    return value

