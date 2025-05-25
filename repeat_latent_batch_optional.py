import torch
import math
from comfy.comfy_types import IO

class RepeatLatentBatchOptional:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to repeat. If None, returns None."}),
                "amount": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of times to repeat the latent."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "repeat"
    CATEGORY = "latent/batch"

    def repeat(self, samples, amount):
        if samples is None:
            return (None,)

        s = samples.copy()
        s_in = samples["samples"]
        s["samples"] = s_in.repeat((amount, 1, 1, 1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount, 1, 1, 1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

NODE_CLASS_MAPPINGS = {
    "RepeatLatentBatchOptional": RepeatLatentBatchOptional
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RepeatLatentBatchOptional": "Repeat Latent Batch (Optional)"
}