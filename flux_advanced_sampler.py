import comfy.samplers
import comfy.sample
import latent_preview
import torch
import node_helpers
from comfy.comfy_types import IO

class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class FluxAdvancedSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model used for sampling."}),
                "conditioning": ("CONDITIONING", {"tooltip": "The conditioning to guide the sampling process."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The name of the sampler to use for Flux sampling."}),
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The random seed for noise generation."
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of sampling steps."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler for sigma calculation."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance scale for conditioning."}),
                "latent_image": ("LATENT", {"tooltip": "The input latent image to sample."}),
                "vae": (IO.VAE, {"tooltip": "The VAE model used to decode the latent image."}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded image from the sampled latent.",)
    FUNCTION = "sample_and_decode"

    CATEGORY = "sampling/flux"
    DESCRIPTION = "Samples a latent image using Flux and decodes it to an image using a VAE."

    def sample_and_decode(self, model, conditioning, sampler_name, noise_seed, steps, denoise, scheduler, guidance, latent_image, vae):
        # Modificar condicionamiento con guidance
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})

        # Crear guider con Guider_Basic
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)

        # Obtener sampler
        sampler = comfy.samplers.sampler_object(sampler_name)

        # Generar ruido
        noise = comfy.sample.prepare_noise(latent_image["samples"], noise_seed, latent_image.get("batch_index", None))

        # Calcular sigmas
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                sigmas = torch.FloatTensor([])
            else:
                total_steps = int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

        # Realizar muestreo
        latent = latent_image.copy()
        latent_image_samples = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])
        latent["samples"] = latent_image_samples
        noise_mask = latent.get("noise_mask", None)
        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise, latent_image_samples, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        # Decodificar el latente con VAE
        decoded_samples = vae.decode(samples.to(vae.vae_dtype))

        return (decoded_samples,)

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "FluxAdvancedSampler": FluxAdvancedSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAdvancedSampler": "Flux Advanced Sampler"
}