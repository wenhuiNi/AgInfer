from pathlib import Path
import sys
from aginfer.aim import AimWriter, Compatibility, VariantPayload
from aginfer.schema import CudaArch, Platform
from tests.executable_fixtures import executable_fixture, WEIGHT_BYTES
from tests.helpers import fake_cubin

plan = executable_fixture()
AimWriter.write(Path(sys.argv[1]), platform=Platform.LINUX_X86_64_GNU,
    manifest={"kind": "synthetic-contract"}, graph={}, tensors={"count": len(plan.values)},
    compatibility=Compatibility(cuda_driver_min=12000), variants=(VariantPayload(CudaArch.SM120, fake_cubin("sm120"),
        WEIGHT_BYTES + bytes(plan.weights_bytes - len(WEIGHT_BYTES)), plan.data),))
