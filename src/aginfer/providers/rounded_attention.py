"""Explicit correctness-first BF16 attention form; no implicit fallback."""
from dataclasses import dataclass
import struct
from ..errors import FormatError, ValidationError
from ..schema import CudaArch

ROUNDED_ATTENTION_PROVIDER_ID = 5
ROUNDED_ATTENTION_PAYLOAD = struct.Struct('<8sHH7IQ32s48s')
assert ROUNDED_ATTENTION_PAYLOAD.size == 128
ROUNDED_MATMUL_PAYLOAD = struct.Struct('<8sHH7IQ32sI4xQ18i24s')
assert ROUNDED_MATMUL_PAYLOAD.size == 192

@dataclass(frozen=True, slots=True)
class RoundedAttentionPayload:
    target_arch: CudaArch
    variant: int  # 1/2: BF16; 3/4: F32 BSHD with TF32/FP32 compute.
    module_bytes: int
    module_sha256: str
    cublaslt_version: int = 0
    qk_algorithm: tuple[int, ...] = ()
    pv_algorithm: tuple[int, ...] = ()
    softmax_warps: int = 1

    def __post_init__(self):
        if (type(self.softmax_warps) is not int or self.softmax_warps not in (1,4)
                or (self.softmax_warps==4 and (not self.cublaslt_version or self.variant not in (1,2)))):
            raise ValidationError('grouped softmax requires materialized BF16 attention')
        if not isinstance(self.target_arch,CudaArch) or self.target_arch != CudaArch.SM120 or type(self.variant) is not int or self.variant not in (1,2,3,4) or (self.variant>=3 and not self.cublaslt_version):
            raise ValidationError('rounded attention requires a delivered SM120 variant')
        if type(self.module_bytes) is not int or not 0 < self.module_bytes < 2**64:
            raise ValidationError('rounded attention module size is invalid')
        if not isinstance(self.module_sha256,str) or len(self.module_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.module_sha256) or not any(c!='0' for c in self.module_sha256):
            raise ValidationError('rounded attention module digest is invalid')
        if type(self.cublaslt_version) is not int or not 0 <= self.cublaslt_version < 2**32:
            raise ValidationError('rounded attention library version is invalid')
        for algorithm in (self.qk_algorithm, self.pv_algorithm):
            if not isinstance(algorithm, tuple) or len(algorithm) != (9 if self.cublaslt_version else 0) or any(type(x) is not int or not 0 <= x < 2**31 for x in algorithm):
                raise ValidationError('rounded attention requires complete fixed algorithms')
            if any(x >= 2**16 for x in algorithm[7:]):
                raise ValidationError('rounded attention inner/cluster algorithm fields overflow')

    @property
    def query_length(self): return {1:50,2:968,3:256,4:256}[self.variant]
    @property
    def key_length(self): return {1:1018,2:968,3:256,4:256}[self.variant]
    @property
    def query_heads(self): return 16 if self.variant>=3 else 8
    @property
    def kv_heads(self): return 16 if self.variant>=3 else 1
    @property
    def head_dim(self): return 72 if self.variant>=3 else 256

    @property
    def workspace_bytes(self):
        return self.query_heads * self.query_length * self.key_length * (4 if self.variant>=3 else 2) if self.cublaslt_version else 0

    def to_bytes(self):
        if self.cublaslt_version:
            grouped=self.softmax_warps==4
            return ROUNDED_MATMUL_PAYLOAD.pack(b'AIRAT3\0\0' if grouped else b'AIRAT2\0\0',3 if grouped else 2,0,int(self.target_arch),self.variant,
                self.query_length,self.key_length,self.query_heads,self.kv_heads,self.head_dim,self.module_bytes,bytes.fromhex(self.module_sha256),
                self.cublaslt_version,self.workspace_bytes,*self.qk_algorithm,*self.pv_algorithm,bytes(24))
        return ROUNDED_ATTENTION_PAYLOAD.pack(b'AIRAT1\0\0',1,0,int(self.target_arch),self.variant,
            self.query_length,self.key_length,8,1,256,self.module_bytes,bytes.fromhex(self.module_sha256),bytes(48))

    @classmethod
    def from_bytes(cls,data):
        if len(data) not in (128,192): raise FormatError('rounded attention payload size is invalid')
        fields=(ROUNDED_MATMUL_PAYLOAD if len(data)==192 else ROUNDED_ATTENTION_PAYLOAD).unpack(data)
        extras=(fields[12],tuple(fields[14:23]),tuple(fields[23:32]),4 if fields[0]==b'AIRAT3\0\0' else 1) if len(data)==192 else ()
        try: parsed=cls(CudaArch(fields[3]),fields[4],fields[10],fields[11].hex(),*extras)
        except (ValueError,ValidationError) as e: raise FormatError('rounded attention payload is invalid') from e
        if parsed.to_bytes()!=bytes(data): raise FormatError('rounded attention payload is noncanonical or unsupported')
        return parsed
