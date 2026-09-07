import dataclasses
import unittest
from aginfer.errors import FormatError, ValidationError
from aginfer.providers.rounded_attention import RoundedAttentionPayload
from aginfer.schema import CudaArch

class RoundedAttentionPayloadTests(unittest.TestCase):
    def test_library_form_round_trip(self):
        for variant in (1, 2, 3, 4):
            p=RoundedAttentionPayload(CudaArch.SM120,variant,80000,'8'*64,
                120803,(21,11,1,0,0,0,13,0,0),(21,19,1,0,0,0,10,0,0))
            self.assertEqual(len(p.to_bytes()),192)
            self.assertEqual(RoundedAttentionPayload.from_bytes(memoryview(p.to_bytes())),p)
            self.assertEqual(p.workspace_bytes,{1:814400,2:14992384,3:4194304,4:4194304}[variant])
            for pos in (0,8,10,20,84,88,92,126,162,168,191):
                changed=bytearray(p.to_bytes());changed[pos]^=1
                with self.assertRaises(FormatError):RoundedAttentionPayload.from_bytes(changed)

    def test_vision_form_is_explicit_f32_bshd(self):
        p=RoundedAttentionPayload(CudaArch.SM120,3,80000,'8'*64,
            120803,(21,15,1,0,0,0,6,0,0),(21,15,1,0,0,0,12,0,0))
        self.assertEqual((p.query_length,p.key_length,p.query_heads,p.kv_heads,p.head_dim),(256,256,16,16,72))
        with self.assertRaises(ValidationError):dataclasses.replace(p,cublaslt_version=0,qk_algorithm=(),pv_algorithm=())

    def test_library_form_requires_complete_algorithms(self):
        p=RoundedAttentionPayload(CudaArch.SM120,1,80000,'8'*64,
            120803,(21,11,1,0,0,0,13,0,0),(21,19,1,0,0,0,10,0,0))
        for kwargs in ({'cublaslt_version':0},{'cublaslt_version':True},
                       {'qk_algorithm':()},{'pv_algorithm':(0,)*8},
                       {'qk_algorithm':(0,)*7+(65536,0)},
                       {'pv_algorithm':(-1,)+(0,)*8}):
            with self.assertRaises(ValidationError):dataclasses.replace(p,**kwargs)

    def test_exact_variants_round_trip(self):
        for variant in (1,2):
            p=RoundedAttentionPayload(CudaArch.SM120,variant,80000,'8'*64)
            self.assertEqual(len(p.to_bytes()),128)
            self.assertEqual(RoundedAttentionPayload.from_bytes(memoryview(p.to_bytes())),p)
            self.assertEqual(p.query_length,50 if variant==1 else 968)
            self.assertEqual(p.key_length,1018 if variant==1 else 968)

    def test_rejects_noncanonical_payload(self):
        blob=RoundedAttentionPayload(CudaArch.SM120,1,80000,'8'*64).to_bytes()
        for position in (0,8,10,12,16,20,24,28,32,36,80,127):
            changed=bytearray(blob);changed[position]^=1
            with self.assertRaises(FormatError): RoundedAttentionPayload.from_bytes(changed)
        for changed in (blob[:-1],blob+b'\0',blob[:48]+bytes(32)+blob[80:]):
            with self.assertRaises(FormatError): RoundedAttentionPayload.from_bytes(changed)

    def test_rejects_unavailable_contracts(self):
        p=RoundedAttentionPayload(CudaArch.SM120,1,80000,'8'*64)
        for kwargs in ({'target_arch':CudaArch.SM89},{'target_arch':120},{'variant':True},
                       {'variant':3},{'module_bytes':0},{'module_bytes':2**64},
                       {'module_sha256':'0'*64},{'module_sha256':'X'*64}):
            with self.assertRaises(ValidationError): dataclasses.replace(p,**kwargs)
