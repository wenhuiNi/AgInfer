from pathlib import Path
import sys
from aginfer.providers.gelu_mul import GeluMulPayload, GeluMulProblem
from aginfer.schema import CudaArch

Path(sys.argv[1]).write_bytes(GeluMulPayload(GeluMulProblem(CudaArch.SM120, 204800), 64000, '6' * 64).to_bytes())
