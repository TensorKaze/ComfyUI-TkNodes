from .multi_latent_selector import NODE_CLASS_MAPPINGS as MultiLatentSelectorMappings
from .multi_latent_selector import NODE_DISPLAY_NAME_MAPPINGS as MultiLatentSelectorDisplayNames
from .flux_latent_sampler import NODE_CLASS_MAPPINGS as FluxLatentSamplerMappings
from .flux_latent_sampler import NODE_DISPLAY_NAME_MAPPINGS as FluxLatentSamplerDisplayNames
from .flux_advanced_sampler import NODE_CLASS_MAPPINGS as FluxAdvancedSamplerMappings
from .flux_advanced_sampler import NODE_DISPLAY_NAME_MAPPINGS as FluxAdvancedSamplerDisplayNames
from .multi_model_loader import NODE_CLASS_MAPPINGS as MultiModelLoaderMappings
from .multi_model_loader import NODE_DISPLAY_NAME_MAPPINGS as MultiModelLoaderDisplayNames
from .load_image_and_scale import NODE_CLASS_MAPPINGS as LoadImageAndScaleMappings
from .load_image_and_scale import NODE_DISPLAY_NAME_MAPPINGS as LoadImageAndScaleDisplayNames
from .load_model_and_upscale import NODE_CLASS_MAPPINGS as LoadModelAndUpscaleMappings
from .load_model_and_upscale import NODE_DISPLAY_NAME_MAPPINGS as LoadModelAndUpscaleDisplayNames
from .vae_encode_optional import NODE_CLASS_MAPPINGS as VAEEncodeOptionalMappings
from .vae_encode_optional import NODE_DISPLAY_NAME_MAPPINGS as VAEEncodeOptionalDisplayNames
from .repeat_latent_batch_optional import NODE_CLASS_MAPPINGS as RepeatLatentBatchOptionalMappings
from .repeat_latent_batch_optional import NODE_DISPLAY_NAME_MAPPINGS as RepeatLatentBatchOptionalDisplayNames

NODE_CLASS_MAPPINGS = {
    **MultiLatentSelectorMappings,
    **FluxLatentSamplerMappings,
    **MultiModelLoaderMappings,
    **FluxAdvancedSamplerMappings,
    **LoadImageAndScaleMappings,
    **LoadModelAndUpscaleMappings,
    **VAEEncodeOptionalMappings,
    **RepeatLatentBatchOptionalMappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MultiLatentSelectorDisplayNames,
    **FluxLatentSamplerDisplayNames,
    **MultiModelLoaderDisplayNames,
    **FluxAdvancedSamplerDisplayNames,
    **LoadImageAndScaleDisplayNames,
    **LoadModelAndUpscaleDisplayNames,
    **VAEEncodeOptionalDisplayNames,
    **RepeatLatentBatchOptionalDisplayNames
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']