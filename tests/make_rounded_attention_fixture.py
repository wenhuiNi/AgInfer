from pathlib import Path
import sys
from aginfer.providers.rounded_attention import RoundedAttentionPayload
from aginfer.schema import CudaArch
Path(sys.argv[1]).write_bytes(b''.join(
    RoundedAttentionPayload(CudaArch.SM120,variant,80000,'8'*64).to_bytes()
    for variant in (1,2))+b''.join(
    RoundedAttentionPayload(CudaArch.SM120,variant,80000,'8'*64,120803,
        (21,11,1,0,0,0,13,0,0),(21,19,1,0,0,0,10,0,0)).to_bytes()
    for variant in (1,2,3,4)))
