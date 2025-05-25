import torch
from comfy.comfy_types import IO

class VAEEncodeOptional:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The VAE model used for encoding the image to latent space."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The image to encode to latent space. If not provided, returns None."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The encoded latent image, or None if no image is provided.",)
    FUNCTION = "encode"

    CATEGORY = "latent"
    DESCRIPTION = "Encodes an image to latent space using a VAE model. If no image is provided, acts as a bypass and returns None."

    def encode(self, vae, image=None):
        # Modo bypass: si no hay imagen, devolver None
        if image is None:
            return (None,)

        # Codificar la imagen con el VAE
        try:
            # Asegurarse de que la imagen solo use los canales RGB (ignorar alfa si existe)
            latent = vae.encode(image[:,:,:,:3])
            return ({"samples": latent},)
        except Exception as e:
            # En caso de error (por ejemplo, dimensiones inválidas), devolver None
            return (None,)

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "VAEEncodeOptional": VAEEncodeOptional
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEEncodeOptional": "VAE Encode (Optional)"
}