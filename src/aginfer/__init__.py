"""AgInfer AOT compiler package."""

from .aim import AimReader, AimWriter, Compatibility, ProviderRequirement
from .constant_store import ConstantCoverage, ConstantKey, ConstantStore
from .frontend import FrontendOutput, SourceFrontend
from .lowering import LoweringInventory, build_lowering_inventory, dump_lowering_inventory
from .processor_assets import SourceAssetContract, inspect_source_asset_contract
from .safetensors import SafetensorsReader
from .schema import CudaArch, Platform
from .source_manifest import SourceManifest, build_source_manifest
from .source_package import AssetBundle, SourcePackage, open_source_package

__all__ = [
    "AimReader",
    "AimWriter",
    "AssetBundle",
    "Compatibility",
    "ConstantCoverage",
    "ConstantKey",
    "ConstantStore",
    "CudaArch",
    "FrontendOutput",
    "LoweringInventory",
    "Platform",
    "ProviderRequirement",
    "SafetensorsReader",
    "SourceManifest",
    "SourceFrontend",
    "SourceAssetContract",
    "SourcePackage",
    "build_source_manifest",
    "build_lowering_inventory",
    "dump_lowering_inventory",
    "open_source_package",
    "inspect_source_asset_contract",
]
__version__ = "0.1.0"
