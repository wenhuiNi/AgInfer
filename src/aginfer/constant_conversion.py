"""Endian-explicit exact BF16 to F32 constant expansion, with bounded buffers."""
from .errors import ValidationError


def widen_bf16(data):
    if len(data) % 2:
        raise ValidationError('BF16 constant chunk is not element aligned')
    output = bytearray(len(data) * 2)
    output[2::4] = data[0::2]
    output[3::4] = data[1::2]
    return bytes(output)


def cast_chunks(constants, source, target, source_bytes, *, chunk_size):
    if (source.constant_identity is None or ':' not in source.constant_identity
            or source.type.dtype != 'bf16' or target.type.dtype != 'f32'
            or source.type.shape != target.type.shape):
        raise ValidationError('invalid constant BF16 widening source/target')
    namespace, name = source.constant_identity.split(':', 1)
    tensor = constants.tensor(namespace, name)
    if (tensor.dtype != 'BF16' or tuple(tensor.shape) != source.type.shape
            or tensor.byte_length != source_bytes):
        raise ValidationError('constant widening source contract changed')
    count, pending = 0, b''
    # Also support stores that split an element at a chunk boundary.
    for block in constants.iter_chunks(namespace, name, chunk_size=max(2, chunk_size // 2)):
        count += len(block)
        if count > source_bytes:
            raise ValidationError('constant widening source exceeds declared size')
        block = pending + block
        end = len(block) - len(block) % 2
        if end:
            yield widen_bf16(block[:end])
        pending = block[end:]
    if pending or count != source_bytes:
        raise ValidationError('constant widening source is short or unaligned')
