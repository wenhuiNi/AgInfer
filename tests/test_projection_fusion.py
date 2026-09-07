from dataclasses import replace
import unittest

from aginfer.compiler.projection_fusion import fuse_projections, FusionConstantView, DerivedTensor, NAMESPACE
from aginfer.compiler.verify import decode_command_payload
from aginfer.errors import FormatError, ValidationError
from aginfer.ir import DType, Device, Function, Op, Program, Region, TensorType, Value, attributes, dump_program
from aginfer.lowering import build_execution_schedule, placements_from_partial_lowering
from aginfer.providers.projection_split import ProjectionSplitPayload, ProjectionSplitProblem, lower_projection_splits
from aginfer.schema import CudaArch


def projection_program(rows=50, widths=(32, 8, 8), bias=0.0, dtype=DType.BF16, name='main'):
    def tensor(shape):
        return TensorType(dtype, shape, device=Device.CUDA)
    ops = []
    for i, width in enumerate(widths):
        ops.extend((
            Op('constant_ref', (), (Value(f'w{i}', tensor((width, 16))),), attributes(namespace='weights', name=f'w{i}')),
            Op('constant', (), (Value(f'b{i}', tensor((width,))),), attributes(value=(bias,) * width)),
            Op('linear', ('x', f'w{i}', f'b{i}'), (Value(f'y{i}', tensor((1, rows, width))),)),
        ))
    return Program((Function(name, (Value('x', tensor((1, rows, 16))),),
                             tuple(f'y{i}' for i in range(len(widths))), Region(tuple(ops))),), name)


class ProjectionFusionTests(unittest.TestCase):
    def test_shared_input_fusion_preserves_outputs_and_is_deterministic(self):
        for name, widths in (('decoder', (32, 8, 8)), ('diffusion', (16, 16, 16))):
            p = projection_program(name=name, widths=widths)
            fused = fuse_projections(p)
            self.assertEqual(len(fused.groups), 1)
            self.assertEqual(fused.program.functions[0].outputs, p.functions[0].outputs)
            self.assertEqual(sum(op.opcode == 'linear' for op in fused.program.functions[0].body.ops), 1)
            self.assertEqual(dump_program(fused.program), dump_program(fuse_projections(p).program))
            with self.assertRaises(ValidationError):
                fuse_projections(fused.program)

    def test_negative_envelopes_preserve_program(self):
        for kwargs in ({'rows': 256}, {'dtype': DType.F32}, {'bias': 1.0}, {'bias': -0.0},
                       {'widths': (8, 8)}, {'widths': (8, 8, 8, 8)}, {'widths': (7, 8, 8)}):
            with self.subTest(kwargs=kwargs):
                p = projection_program(**kwargs)
                self.assertEqual(fuse_projections(p).program, p)

    def test_weight_packing_streams_original_bytes_in_channel_order(self):
        fused = fuse_projections(projection_program())
        class Source:
            namespaces = ('weights',)
            def tensor(self, namespace, name):
                width = (32, 8, 8)[int(name[-1])]
                return DerivedTensor('BF16', (width, 16), width * 32)
            def iter_chunks(self, namespace, name, *, chunk_size):
                data = bytes([int(name[-1]) + 1]) * self.tensor(namespace, name).byte_length
                for offset in range(0, len(data), chunk_size):
                    yield data[offset:offset + chunk_size]
        view = FusionConstantView(Source(), fused.constants)
        weight = fused.groups[0]['weight']
        self.assertEqual(b''.join(view.iter_chunks(NAMESPACE, weight, chunk_size=17)),
                         bytes([1]) * 1024 + bytes([2]) * 256 + bytes([3]) * 256)
        for name, spec in fused.constants.items():
            if spec.get('zero'):
                self.assertEqual(b''.join(view.iter_chunks(NAMESPACE, name, chunk_size=17)), bytes(96))
        with self.assertRaises(ValidationError):
            view.tensor(NAMESPACE, 'invalid')

    def test_separate_inputs_and_non_source_weights_are_not_fused(self):
        p = projection_program()
        f = p.functions[0]
        for separate_input in (True, False):
            ops = list(f.body.ops)
            inputs = f.inputs
            if separate_input:
                inputs += (replace(inputs[0], value_id='other'),)
                ops[-1] = replace(ops[-1], inputs=('other', 'w2', 'b2'))
            else:
                ops[0] = replace(ops[0], opcode='constant', attributes=attributes(value=(0.0,) * (32 * 16)))
            changed = replace(p, functions=(replace(f, inputs=inputs, body=Region(tuple(ops))),))
            self.assertEqual(fuse_projections(changed).program, changed)

    def test_split_requires_exclusive_complete_partition(self):
        p = fuse_projections(projection_program()).program
        f = p.functions[0]
        merged = next(op.outputs[0].value_id for op in f.body.ops if op.opcode == 'linear')
        exported = replace(p, functions=(replace(f, outputs=f.outputs + (merged,)),))
        self.assertFalse(lower_projection_splits(build_execution_schedule(exported), 64000, '6' * 64).commands)
        ops = tuple(op for op in f.body.ops if not (op.opcode == 'slice' and op.outputs[0].value_id == 'y2'))
        partial = replace(p, functions=(replace(f, outputs=f.outputs[:2], body=Region(ops)),))
        self.assertFalse(lower_projection_splits(build_execution_schedule(partial), 64000, '6' * 64).commands)

    def test_split_lowering_covers_three_slices_in_one_command(self):
        schedule = build_execution_schedule(fuse_projections(projection_program()).program)
        split = lower_projection_splits(schedule, 64000, '6' * 64)
        self.assertEqual(len(split.commands), 1)
        command = split.commands[0]
        self.assertEqual(len(command.fused_execution_indices), 3)
        self.assertEqual(len(command.command.operands), 4)
        self.assertEqual(command.command.workspace_bytes, 0)
        self.assertFalse(command.command.capture_safe)
        self.assertEqual(decode_command_payload(command.command).problem.widths, (32, 8, 8))
        self.assertEqual(len(placements_from_partial_lowering(schedule, split)), 1)

    def test_split_payload_roundtrip_and_rejections(self):
        p = ProjectionSplitPayload(ProjectionSplitProblem(CudaArch.SM120, 50, (2048, 256, 256)), 64000, '6' * 64)
        data = p.to_bytes()
        self.assertEqual(len(data), 128)
        self.assertEqual(ProjectionSplitPayload.from_bytes(data), p)
        for offset in (0, 8, 10, 12, 16, 20, 72, 127):
            corrupted = bytearray(data)
            corrupted[offset] ^= 1 if offset == 20 else 128
            with self.subTest(offset=offset), self.assertRaises(FormatError):
                ProjectionSplitPayload.from_bytes(corrupted)
        for rows, widths in ((0, (8, 8, 8)), (129, (8, 8, 8)), (1, (7, 8, 8)), (1, (65536, 8, 8))):
            with self.assertRaises(ValidationError):
                replace(p, problem=ProjectionSplitProblem(CudaArch.SM120, rows, widths))
