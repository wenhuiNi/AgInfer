from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .constant_store import ConstantCoverage, ConstantKey, ConstantStore, CoverageRecord
from .errors import ValidationError
from .ir import DType, Program, verify_program
from .source_package import AssetBundle, SourcePackage


@dataclass(frozen=True, slots=True)
class FrontendOutput:
    """Framework-free result handed from a source frontend to IR passes."""

    program: Program
    constants: ConstantStore
    assets: AssetBundle
    coverage: tuple[CoverageRecord, ...]

    @classmethod
    def finalize(
        cls,
        program: Program,
        source: SourcePackage,
        coverage: ConstantCoverage,
    ) -> "FrontendOutput":
        verify_program(program)
        if coverage.store is not source.constants:
            raise ValidationError("frontend coverage does not belong to the source constant store")
        coverage.require_complete()
        records = coverage.records
        dispositions = {record.key: record.disposition for record in records}
        dtype_map = {"F32": DType.F32, "F16": DType.F16, "BF16": DType.BF16}
        for function in program.functions:
            for op in function.body.ops:
                if op.opcode != "constant_ref":
                    continue
                namespace = str(op.attribute("namespace"))
                name = str(op.attribute("name"))
                tensor = source.constants.tensor(namespace, name)
                key = ConstantKey(namespace, name)
                if dispositions[key] != "consumed":
                    raise ValidationError(f"ProgramIR references an ignored source constant: {namespace}:{name}")
                dtype = dtype_map.get(tensor.dtype)
                if dtype is None:
                    raise ValidationError(f"ProgramIR cannot represent source constant dtype {tensor.dtype}: {name}")
                output_type = op.outputs[0].type
                if output_type.dtype != dtype or output_type.shape != tuple(tensor.shape):
                    raise ValidationError(f"ProgramIR constant_ref type disagrees with source: {namespace}:{name}")
        return cls(program, source.constants, source.assets, records)


class SourceFrontend(Protocol):
    """Structural importer contract; implementations live outside core IR."""

    frontend_id: str

    def import_program(self, source: SourcePackage) -> FrontendOutput:
        ...
