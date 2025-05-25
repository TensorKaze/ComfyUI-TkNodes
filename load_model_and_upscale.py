import os
import torch
import folder_paths
import comfy
import comfy.utils
from comfy.comfy_types import IO
from spandrel import ModelLoader

class LoadModelAndUpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to upscale."}),
                "model_name": (folder_paths.get_filename_list("upscale_models"), {"tooltip": "The upscale model to use."}),
                "bypass_upscaler": ("BOOLEAN", {"default": False, "toggle": True, "label_on": "yes", "label_off": "no", "tooltip": "Bypass upscaling and return the original image if yes."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_and_upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Loads an upscale model and upscales the input image using the model, unless bypass_upscaler is yes."

    def load_and_upscale(self, image, model_name, bypass_upscaler):
        # Si bypass_upscaler es True ("yes"), devolver la imagen original
        if bypass_upscaler:
            return (image,)

        # Cargar el modelo de escalado (exactamente como UpscaleModelLoader en nodes_upscale_model.py)
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        upscale_model = ModelLoader().load_from_state_dict(sd).eval()

        # Escalar la imagen con el modelo (exactamente como ImageUpscaleWithModel en nodes_upscale_model.py)
        device = comfy.model_management.get_torch_device()
        memory_required = comfy.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        comfy.model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)  # (B, H, W, C) -> (B, C, H, W)

        tile = 512
        overlap = 32
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        scaled_image = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # (B, C, H', W') -> (B, H', W', C)

        # Devolver la imagen escalada
        return (scaled_image,)

    @classmethod
    def IS_CHANGED(cls, image, model_name, bypass_upscaler, **kwargs):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        return comfy.utils.calculate_file_hash(model_path)

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "LoadModelAndUpscaleImage": LoadModelAndUpscaleImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadModelAndUpscaleImage": "Load Model & Upscale Image"
}