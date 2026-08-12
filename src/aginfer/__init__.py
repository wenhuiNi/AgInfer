"""AgInfer AOT compiler package."""

from .aim import AimReader, AimWriter
from .schema import CudaArch, Platform

__all__ = ["CudaArch", "AimReader", "AimWriter", "Platform"]
__version__ = "0.1.0"
