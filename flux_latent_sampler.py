import torch
import comfy.model_management
import comfy.model_sampling
import nodes

class FluxLatentSampler:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "max_shift": ("FLOAT", {"default": 1.15, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "sample"
    CATEGORY = "advanced/model"

    def sample(self, model, max_shift, base_shift, width, height, batch_size):
        # Generar el latente vacío
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        latent_output = {"samples": latent}

        # Ajustar parámetros de muestreo para Flux
        m = model.clone()

        # Calcular shift
        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        # Crear la clase de muestreo
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)

        return (m, latent_output)

NODE_CLASS_MAPPINGS = {
    "FluxLatentSampler": FluxLatentSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLatentSampler": "Flux Latent Sampler"
}