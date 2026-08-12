from __future__ import annotations

from .base import Adapter, AdapterResult, register
from ..checkpoint import Checkpoint


@register
class LingBotVla1Adapter(Adapter):
    name = "lingbot_vla_1"

    def build(self, checkpoint: Checkpoint, profile: dict) -> AdapterResult:
        self._validate_model_type(checkpoint, ("lingbot", "qwen2_5_vl", "qwen2.5-vl", "pi0"))
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
            "subgraphs": ["qwen2_5_vision", "vlm_conditioning", "pi_action_expert_step"],
            "recipe": {
                "depth_input": False,
                "conditioning": {"execute": "once", "cache": True},
                "action_head": {"execute": "denoising_loop", "schedule": "profile"},
            },
        }
        return AdapterResult(
            "lingbot_vla_1_no_depth",
            self.version,
            graph,
            [
                {"name": "input_ids", "dtype": "I32", "location": "device"},
                {"name": "pixel_values", "dtype": "checkpoint", "location": "device"},
                {"name": "state", "dtype": "checkpoint", "location": "device"},
            ],
            [{"name": "actions", "dtype": "checkpoint", "location": "device"}],
        )

