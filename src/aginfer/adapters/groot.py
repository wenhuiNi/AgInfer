from __future__ import annotations

from .base import Adapter, AdapterResult, register
from ..checkpoint import Checkpoint


@register
class GrootN15Adapter(Adapter):
    name = "groot_n1_5"

    def build(self, checkpoint: Checkpoint, profile: dict) -> AdapterResult:
        self._validate_model_type(checkpoint, ("groot", "eagle"))
        graph = {
            "opset": 1,
            "passes": [
                "constant_fold",
                "remove_training_nodes",
                "eliminate_view_ops",
                "fuse_qkv",
                "fuse_mlp",
                "fuse_norm",
                "deduplicate_tied_weights",
            ],
            "subgraphs": ["vision", "vlm_conditioning", "flow_matching_dit_step"],
            "recipe": {
                "conditioning": {"execute": "once", "cache": True},
                "action_head": {"execute": "denoising_loop", "schedule": "profile"},
            },
        }
        return AdapterResult(
            "groot_n1_5",
            self.version,
            graph,
            [
                {"name": "input_ids", "dtype": "I32", "location": "device"},
                {"name": "pixel_values", "dtype": "checkpoint", "location": "device"},
                {"name": "state", "dtype": "checkpoint", "location": "device"},
            ],
            [{"name": "actions", "dtype": "checkpoint", "location": "device"}],
        )

