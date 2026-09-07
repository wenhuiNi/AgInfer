"""Tiny real-provider graph fixture; no model weights or numerical receipts."""
import hashlib
from pathlib import Path
import sys

from aginfer.aim import AimWriter, Compatibility, VariantPayload
from aginfer.executable import compile_executable_plan
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value
from aginfer.lowering import (CommandOperand, CommandStream, CommandTag, OperandAccess,
    ProviderCommand, build_execution_schedule, build_memory_plan, dump_memory_plan)
from aginfer.providers import AotPointwiseProblem, CudaKernelDType, CudaKernelId, CudaKernelPayload
from aginfer.schema import CudaArch, Platform

module = Path(sys.argv[1]).read_bytes()
tensor = TensorType(DType.F32, (256,), device=Device.CUDA)
ops = (
    Op("add", ("x", "y"), (Value("tmp", tensor),)),
    Op("add", ("tmp", "y"), (Value("out", tensor),)),
)
function = Function("main", (Value("x", tensor), Value("y", tensor)), ("out",), Region(ops))
program = Program(functions=(function,), entry="main")
schedule = build_execution_schedule(program)
memory = build_memory_plan(schedule)
problem = AotPointwiseProblem(CudaArch.SM120, CudaKernelId.ADD_F32, "add", CudaKernelDType.F32, 256)
for bad, path in enumerate(sys.argv[2:]):
    commands = []
    for index, op in enumerate(schedule.ops):
        # Bad second command exercises partial-Prepare cleanup and no fallback.
        digest = hashlib.sha256(module + (b"bad" if bad and index == 1 else b"")).hexdigest()
        payload = CudaKernelPayload.for_problem(problem, module_bytes=len(module), module_sha256=digest).to_bytes()
        commands.append(ProviderCommand(CommandTag.CUDA_KERNEL, 2, 1, 0,
            hashlib.sha256(payload).hexdigest(),
            tuple(CommandOperand(v, OperandAccess.READ) for v in op.inputs) +
            tuple(CommandOperand(v, OperandAccess.WRITE) for v in op.outputs),
            payload, capture_safe=False))
    stream = CommandStream(target_arch=CudaArch.SM120,
        memory_plan_sha256=hashlib.sha256(dump_memory_plan(memory).encode()).hexdigest(),
        value_count=len(schedule.values), arena_bytes=memory.arena_bytes,
        state_bytes=memory.state_bytes, workspace_bytes=0, commands=tuple(commands))
    plan = compile_executable_plan(schedule, memory, stream, (), weights_bytes=256)
    AimWriter.write(Path(path), platform=Platform.LINUX_X86_64_GNU,
        manifest={"kind": "synthetic-cuda-graph-contract"}, graph={}, tensors={},
        compatibility=Compatibility(cuda_driver_min=12000),
        variants=(VariantPayload(CudaArch.SM120, module, bytes(256), plan.data),))
