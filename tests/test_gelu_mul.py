from dataclasses import replace
import unittest

from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes
from aginfer.lowering import build_execution_schedule, build_memory_plan, placements_from_partial_lowering, assemble_command_stream
from aginfer.lowering.memory import AllocationRegion, replan_memory_for_commands
from aginfer.providers.gelu_mul import GeluMulPayload, GeluMulProblem, lower_gelu_mul
from aginfer.schema import CudaArch


def program(shape=(1, 50, 4096), dtype=DType.BF16, exported=False, extra=False, reverse=False, both=False):
    t = TensorType(dtype, shape, device=Device.CUDA)
    ops = [Op('gelu', ('gate',), (Value('g', t),), attributes(approximation='tanh'))]
    other = 'up'
    if both:
        ops.append(Op('gelu', ('up',), (Value('u', t),), attributes(approximation='tanh')))
        other = 'u'
    ops.append(Op('mul', (other, 'g') if reverse else ('g', other), (Value('y', t),)))
    outputs = ('y', 'g') if exported else ('y',)
    if extra:
        ops.append(Op('add', ('g', 'up'), (Value('z', t),)))
        outputs += ('z',)
    return Program((Function('generic_ffn', (Value('gate', t), Value('up', t)), outputs, Region(tuple(ops))),), 'generic_ffn')


class GeluMulTests(unittest.TestCase):
    def test_command_liveness_eliminates_intermediate_storage(self):
        schedule = build_execution_schedule(program())
        result = lower_gelu_mul(schedule, 64000, '6' * 64)
        placements = placements_from_partial_lowering(schedule, result)
        memory = replan_memory_for_commands(schedule, build_memory_plan(schedule), placements)
        self.assertEqual(memory.allocations[schedule.ops[0].outputs[0]].region, AllocationRegion.UNUSED)
        commands = assemble_command_stream(schedule, memory, placements, target_arch=CudaArch.SM120)
        self.assertEqual(len(commands.commands), 1)

    def test_structural_fusion_and_operand_order(self):
        for shape in ((1, 50, 4096), (1, 968, 16384), (37,)):
            for reverse in (False, True):
                schedule = build_execution_schedule(program(shape=shape, reverse=reverse))
                result = lower_gelu_mul(schedule, 64000, '6' * 64)
                self.assertEqual(len(result.commands), 1)
                command = result.commands[0]
                self.assertEqual(len(command.fused_execution_indices), 2)
                self.assertEqual(len(command.command.operands), 3)
                self.assertEqual(len(placements_from_partial_lowering(schedule, result)), 1)
                self.assertEqual(decode_command_payload(command.command).module_bytes, 64000)
                self.assertFalse(command.command.capture_safe)
                self.assertEqual(command.command.workspace_bytes, 0)
                inputs = dict(schedule.entry_inputs)
                self.assertEqual(tuple(x.value_id for x in command.command.operands[:2]), (inputs['gate'], inputs['up']))

    def test_no_exported_shared_or_out_of_contract_intermediate(self):
        for kwargs in ({'exported': True}, {'extra': True}, {'dtype': DType.F32}, {'shape': (1 << 27,)}):
            self.assertFalse(lower_gelu_mul(build_execution_schedule(program(**kwargs)), 1, '1' * 64).commands)
        p = program()
        f = p.functions[0]
        ops = (replace(f.body.ops[0], attributes=attributes(approximation='erf')), f.body.ops[1])
        # Alter the already-built schedule: the source IR verifier independently
        # refuses unsupported approximations, and lowering must refuse them too.
        s = build_execution_schedule(p)
        s = replace(s, ops=(replace(s.ops[0], attributes=ops[0].attributes), s.ops[1]))
        self.assertFalse(lower_gelu_mul(s, 1, '1' * 64).commands)

    def test_two_activated_inputs_do_not_claim_one_mul_twice(self):
        s = build_execution_schedule(program(both=True))
        result = lower_gelu_mul(s, 1, '1' * 64)
        self.assertEqual(len(result.commands), 1)
        self.assertEqual(result.commands[0].fused_execution_indices, (0, 2))

    def test_payload_canonical_and_bounded(self):
        p = GeluMulPayload(GeluMulProblem(CudaArch.SM120, 204800), 64000, '6' * 64)
        data = p.to_bytes()
        self.assertEqual(len(data), 128)
        self.assertEqual(GeluMulPayload.from_bytes(data), p)
        for offset in (0, 8, 10, 12, 23, 64, 127):
            bad = bytearray(data); bad[offset] ^= 128
            with self.subTest(offset=offset), self.assertRaises(FormatError):
                GeluMulPayload.from_bytes(bad)
        for n in (0, -1, True, (1 << 26) + 1):
            with self.assertRaises(ValidationError):
                GeluMulProblem(CudaArch.SM120, n)
        with self.assertRaises(FormatError):
            GeluMulPayload.from_bytes(data[:-1])


if __name__ == '__main__':
    unittest.main()
