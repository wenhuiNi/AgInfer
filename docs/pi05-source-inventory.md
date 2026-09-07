# PI0.5 source inventory (not a model-support claim)

AgInfer's first model-recipe audit targets the LeRobot PI0.5 flow-matching
checkpoint shape. It validates source metadata and tensor identity only; it
does not yet construct the complete ProgramIR, lower providers, or run model
inference.

The recipe requires the currently verified configuration envelope:

- PaliGemma `gemma_2b` and action expert `gemma_300m`;
- BF16 model precision, 224×224 internal image resolution;
- positive static action dimension, chunk length, and denoise-step count.

For the validation checkpoint, the exact source inventory is:

| Region | Tensors | Disposition |
|---|---:|---|
| Vision patch/position, 27 encoder layers, post norm | 437 | consumed |
| Multimodal projector | 2 | consumed |
| Language embedding, 18 VLM layers, final norm | 164 | consumed |
| 18-layer conditional action expert and final norm | 200 | consumed |
| Action/time projections | 8 | consumed |
| Expert LM head | 1 | explicitly ignored: inference returns hidden states through the action projection |
| Preprocessor statistics | 90 | consumed by preprocessing asset contract |
| Postprocessor statistics | 90 | consumed by postprocessing asset contract |

That closes all 992 source constants: 991 consumed and one explicitly ignored.
Every model tensor name, source dtype, and shape is checked. Missing, extra, or
changed entries fail the audit.

The checkpoint does not contain tokenizer files. Its processor contract names
`google/paligemma-3b-pt-224` as an external requirement. AgInfer reports this
requirement and does not download or execute it implicitly.

The source analysis was cross-checked against the PI0.5 prefix-cache and
fixed-step denoise boundaries in LeRobot revision
`7e241bd630a3719a56157a497ce5d08f244784f1`, as exercised by the separate
reference harness at revision `7f56e1ab420d4dbe1b0c682019bcb1cc98351952`.
Neither repository is imported by AgInfer or referenced by its tests.

Run the audit against a local checkpoint without loading tensor payloads:

```python
from aginfer import open_source_package
from aginfer.recipes.pi05 import audit_pi05_source

source = open_source_package("/path/to/checkpoint", offline=True)
inventory = audit_pi05_source(source)
assert inventory.tensor_count == 992
```

Remaining work before a support claim includes tokenizer asset closure, the
complete image/prompt/state preprocessing contract, full ProgramIR construction
including 18-layer KV-cache state, provider lowering, numerical comparison on
real held-out inputs, and native C++ deployment execution.
