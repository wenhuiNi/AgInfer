from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..checkpoint import Checkpoint
from ..errors import ValidationError


@dataclass(frozen=True)
class AdapterResult:
    model_family: str
    adapter_version: int
    graph: dict[str, Any]
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]


class Adapter(ABC):
    name: str
    version: int = 1

    @abstractmethod
    def build(self, checkpoint: Checkpoint, profile: dict[str, Any]) -> AdapterResult:
        raise NotImplementedError

    def _validate_model_type(self, checkpoint: Checkpoint, accepted: tuple[str, ...]) -> str:
        model_type = str(checkpoint.config.get("model_type", "")).lower()
        architectures = " ".join(str(item).lower() for item in checkpoint.config.get("architectures", []))
        if not any(token in model_type or token in architectures for token in accepted):
            raise ValidationError(
                f"adapter {self.name} does not match checkpoint model_type={model_type!r}; expected one of {accepted}"
            )
        return model_type


_REGISTRY: dict[str, type[Adapter]] = {}


def register(adapter: type[Adapter]) -> type[Adapter]:
    if adapter.name in _REGISTRY:
        raise RuntimeError(f"duplicate adapter registration: {adapter.name}")
    _REGISTRY[adapter.name] = adapter
    return adapter


def create_adapter(name: str) -> Adapter:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValidationError(f"unknown adapter {name!r}; available: {available}") from exc


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))

