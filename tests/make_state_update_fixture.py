import sys
from pathlib import Path
from aginfer.providers.state_update import StateUpdatePayload, StateUpdateProblem
from aginfer.schema import CudaArch
Path(sys.argv[1]).write_bytes(StateUpdatePayload(
    StateUpdateProblem(CudaArch.SM120,1018*256,968*256,0),64000,'6'*64).to_bytes())
