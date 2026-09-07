import sys
from pathlib import Path
from aginfer.providers.projection_split import ProjectionSplitPayload, ProjectionSplitProblem
from aginfer.schema import CudaArch

Path(sys.argv[1]).write_bytes(ProjectionSplitPayload(
    ProjectionSplitProblem(CudaArch.SM120, 50, (2048, 256, 256)), 64000, '6' * 64).to_bytes())
