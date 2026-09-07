from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import resolve_source
from .constant_store import ConstantStore
from .errors import ValidationError
from .source_manifest import SourceAsset, SourceManifest, build_source_manifest


class AssetBundle:
    """Validated access to manifest-declared source assets.

    The bundle inventories all assets, while bounded byte/JSON reads are aimed
    at configuration and preprocessing metadata. Tensor payloads remain the
    responsibility of ``ConstantStore``.
    """

    def __init__(self, manifest: SourceManifest, *, root: str | Path | None = None) -> None:
        self.manifest = SourceManifest.from_dict(manifest.to_dict())
        selected_root = self.manifest.source if root is None else root
        self.root = Path(selected_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValidationError(f"asset bundle root is not a directory: {self.root}")
        self._assets = {asset.asset_id: asset for asset in self.manifest.assets}
        self._paths: dict[str, Path] = {}
        for asset in self.manifest.assets:
            path = (self.root / asset.path).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise ValidationError(f"source asset escapes asset bundle root: {asset.path}") from exc
            if not path.is_file():
                if asset.required:
                    raise ValidationError(f"required source asset is missing: {asset.path}")
                continue
            actual_size = path.stat().st_size
            if actual_size != asset.byte_size:
                raise ValidationError(
                    f"source asset size changed for {asset.path}: manifest={asset.byte_size}, actual={actual_size}"
                )
            self._paths[asset.asset_id] = path

    def select(self, *, role: str | None = None, namespace: str | None = None) -> tuple[SourceAsset, ...]:
        return tuple(
            asset
            for asset in self.manifest.assets
            if (role is None or asset.role == role) and (namespace is None or asset.namespace == namespace)
        )

    def asset(self, asset_id: str) -> SourceAsset:
        try:
            return self._assets[asset_id]
        except KeyError as exc:
            raise ValidationError(f"source asset not found: {asset_id}") from exc

    def asset_by_path(self, path: str) -> SourceAsset:
        matches = [asset for asset in self.manifest.assets if asset.path == path]
        if len(matches) != 1:
            raise ValidationError(f"source asset path is not declared exactly once: {path}")
        return matches[0]

    def read_bytes(self, asset_id: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValidationError("asset read limit must be a positive integer")
        asset = self.asset(asset_id)
        if asset.byte_size > max_bytes:
            raise ValidationError(
                f"source asset exceeds bounded read limit for {asset.path}: size={asset.byte_size}, limit={max_bytes}"
            )
        path = self._paths.get(asset_id)
        if path is None:
            raise ValidationError(f"optional source asset is unavailable: {asset.path}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValidationError(f"cannot read source asset {asset.path}: {exc}") from exc
        if len(payload) != asset.byte_size:
            raise ValidationError(f"source asset size changed while reading: {asset.path}")
        return payload

    def read_json(self, asset_id: str, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
        asset = self.asset(asset_id)
        try:
            return json.loads(self.read_bytes(asset_id, max_bytes=max_bytes).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"source asset is not valid UTF-8 JSON: {asset.path}") from exc


@dataclass(frozen=True, slots=True)
class SourcePackage:
    manifest: SourceManifest
    constants: ConstantStore
    assets: AssetBundle


def open_source_package(
    source: str,
    *,
    revision: str | None = None,
    offline: bool = False,
) -> SourcePackage:
    """Resolve and inventory one source without deserializing tensor payloads."""

    root, resolved_revision = resolve_source(source, revision, offline)
    manifest = build_source_manifest(root, revision=resolved_revision)
    return SourcePackage(
        manifest=manifest,
        constants=ConstantStore(manifest, root=root),
        assets=AssetBundle(manifest, root=root),
    )
