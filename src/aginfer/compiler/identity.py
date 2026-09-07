"""Path-independent, content-addressed build identities."""
import hashlib
import json
import os
from pathlib import Path

from ..errors import ValidationError


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        before = stream.fileno()
        stat = os.fstat(before)
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
            size += len(block)
        after = os.fstat(before)
    if size != stat.st_size or (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValidationError("build input changed while hashing")
    return {"bytes": size, "sha256": h.hexdigest()}


def compiler_identity():
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): file_identity(path) for path in sorted(root.rglob("*.py"))}
    return {"schema": "aginfer.compiler-source.v1", "files": files, "sha256": digest(files)}


def read_json(path, max_bytes=8 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValidationError("build metadata exceeds size limit")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("duplicate build metadata key")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique,
            parse_constant=lambda x: (_ for _ in ()).throw(ValidationError("non-finite build metadata")))
    except (UnicodeError, ValueError) as exc:
        raise ValidationError("invalid build metadata JSON") from exc
