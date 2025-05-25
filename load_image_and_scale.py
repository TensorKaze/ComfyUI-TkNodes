import os
import numpy as np
import torch
import math
from PIL import Image, ImageSequence
import folder_paths
import comfy
from comfy.comfy_types import IO

class LoadImageAndScaleToTotalPixels:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (["none"] + sorted(files), {"image_upload": True, "tooltip": "The image file to load from the input directory. Select 'none' to skip loading an image."}),
                "upscale_method": (cls.upscale_methods, {"tooltip": "The method used for scaling the image."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 16.0,
                    "step": 0.01,
                    "tooltip": "Target total pixels in megapixels (e.g., 1.0 for 1,048,576 pixels)."
                }),
                "bypass_sttp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, skips scaling and returns the original image and mask."
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK)
    RETURN_NAMES = ("IMAGE", "MASK")
    OUTPUT_TOOLTIPS = (
        "The loaded and scaled image (or original if bypassed), or None if no image is provided.",
        "The mask generated from the alpha channel (scaled to match the image or original if bypassed), or None if no image is provided."
    )
    FUNCTION = "load_and_scale"

    CATEGORY = "image"
    DESCRIPTION = "Loads an image from the input directory, scales it to a specified total number of pixels (unless bypassed), and returns the scaled mask. If 'none' is selected, returns None for both outputs."

    def load_and_scale(self, image, upscale_method, megapixels, bypass_sttp):
        # Si se selecciona "none", devolver None para ambas salidas
        if image == "none":
            return (None, None)

        # Cargar la imagen (exactamente como LoadImage en nodes.py)
        image_path = os.path.join(folder_paths.get_input_directory(), image)
        i = None
        try:
            for img in ImageSequence.Iterator(Image.open(image_path)):
                i = img.convert("RGBA")
                break
            if i is None:
                raise ValueError(f"Failed to load image: {image_path}")
        except Exception:
            return (None, None)

        img = np.array(i).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        
        if img.shape[-1] == 4:
            mask = img[0, :, :, 3].clone()  # (H, W), float32
        else:
            mask = torch.ones((img.shape[1], img.shape[2]), dtype=torch.float32, device=img.device)
        image_tensor = img[:, :, :, :3]  # (1, H, W, 3)

        # Si bypass_sttp está activado, devolver imagen y máscara originales
        if bypass_sttp:
            return (image_tensor, mask)

        # Escalar la imagen
        samples = image_tensor.movedim(-1, 1)  # (1, H, W, 3) -> (1, 3, H, W)
        total = int(megapixels * 1024 * 1024)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        scaled_samples = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        scaled_samples = scaled_samples.movedim(1, -1)  # (1, 3, H', W') -> (1, H', W', 3)

        # Escalar la máscara
        mask_samples = mask[None, None, :, :]  # (H, W) -> (1, 1, H, W)
        # Usar bicubic para la máscara si upscale_method es lanczos, si no, usar el mismo método
        mask_upscale_method = "bicubic" if upscale_method == "lanczos" else upscale_method
        scaled_mask = comfy.utils.common_upscale(mask_samples, width, height, mask_upscale_method, "disabled")
        scaled_mask = scaled_mask[0, 0, :, :]  # (1, 1, H', W') -> (H', W')

        return (scaled_samples, scaled_mask)

    @classmethod
    def IS_CHANGED(cls, image, upscale_method, megapixels, bypass_sttp, **kwargs):
        if image == "none":
            return None
        image_path = os.path.join(folder_paths.get_input_directory(), image)
        try:
            hash_value = comfy.utils.calculate_file_hash(image_path)
            return f"{hash_value}_{bypass_sttp}"
        except AttributeError:
            return f"{image_path}_{bypass_sttp}"

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "LoadImageAndScaleToTotalPixels": LoadImageAndScaleToTotalPixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageAndScaleToTotalPixels": "Load Image & Scale to Total Pixels"
}