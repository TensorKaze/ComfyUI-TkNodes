import torch
import comfy.sd
import comfy.utils
import folder_paths
from comfy.comfy_types import IO, InputTypeDict

class MultiModelLoader:
    def __init__(self):
        # Cache para almacenar modelos y LoRA
        self.cached_model = None
        self.cached_clip = None
        self.cached_vae = None
        self.cached_lora = None
        self.last_unet_name = None
        self.last_weight_dtype = None
        self.last_clip_name1 = None
        self.last_clip_name2 = None
        self.last_clip_type = None
        self.last_clip_device = None
        self.last_vae_name = None
        self.last_lora_name = None

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        loras = ["none"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "The name of the diffusion model (UNET) to load."
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "tooltip": "The weight dtype for the diffusion model."
                }),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "The name of the first CLIP text encoder to load."
                }),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "The name of the second CLIP text encoder to load."
                }),
                "clip_type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], {
                    "tooltip": "The type of CLIP configuration (e.g., flux: clip-l, t5)."
                }),
                "clip_device": (["default", "cpu"], {
                    "advanced": True,
                    "tooltip": "Device for loading CLIP models."
                }),
                "vae_name": (cls.vae_list(), {
                    "tooltip": "The name of the VAE model to load."
                }),
                "lora_name": (loras, {
                    "tooltip": "The name of the LoRA to load (select 'none' to skip loading)."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Strength of the LoRA applied to the diffusion model."
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Strength of the LoRA applied to the CLIP model."
                }),
                "bypass_lora": ("BOOLEAN", {
                    "default": False,
                    "toggle": True,
                    "label_on": "yes",
                    "label_off": "no",
                    "tooltip": "Bypass LoRA application to MODEL and CLIP if yes, but keep it loaded."
                }),
            },
        }

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))
        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]
        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]
        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    RETURN_TYPES = (IO.MODEL, IO.CLIP, IO.VAE)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The diffusion model (modified by LoRA if loaded and not bypassed).",
        "The CLIP model (modified by LoRA if loaded and not bypassed).",
        "The VAE model used for encoding/decoding latents."
    )
    FUNCTION = "load_unified"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model, dual CLIP models, VAE, and optionally applies a loaded LoRA to the model and CLIP with bypass option."

    def load_unified(self, unet_name, weight_dtype, clip_name1, clip_name2, clip_type, clip_device, vae_name, lora_name, strength_model, strength_clip, bypass_lora):
        # Cargar modelo de difusión si no está cacheado o cambió
        if (self.cached_model is None or
                unet_name != self.last_unet_name or
                weight_dtype != self.last_weight_dtype):
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            self.cached_model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            self.last_unet_name = unet_name
            self.last_weight_dtype = weight_dtype

        model = self.cached_model

        # Cargar CLIPs si no están cacheados o cambiaron
        if (self.cached_clip is None or
                clip_name1 != self.last_clip_name1 or
                clip_name2 != self.last_clip_name2 or
                clip_type != self.last_clip_type or
                clip_device != self.last_clip_device):
            clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
            clip_options = {}
            if clip_device == "cpu":
                clip_options["load_device"] = clip_options["offload_device"] = torch.device("cpu")
            self.cached_clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type_enum,
                model_options=clip_options
            )
            self.last_clip_name1 = clip_name1
            self.last_clip_name2 = clip_name2
            self.last_clip_type = clip_type
            self.last_clip_device = clip_device

        clip = self.cached_clip

        # Cargar VAE si no está cacheado o cambió
        if self.cached_vae is None or vae_name != self.last_vae_name:
            if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
                sd = self.load_taesd(vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                sd = comfy.utils.load_torch_file(vae_path)
            self.cached_vae = comfy.sd.VAE(sd=sd)
            self.cached_vae.throw_exception_if_invalid()
            self.last_vae_name = vae_name

        vae = self.cached_vae

        # Cargar LoRA solo si lora_name no es "none" y no está cacheado o cambió
        if lora_name != "none" and (self.cached_lora is None or lora_name != self.last_lora_name):
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            self.cached_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.last_lora_name = lora_name
        elif lora_name == "none":
            self.cached_lora = None
            self.last_lora_name = None

        # Aplicar LoRA solo si está cargado, bypass_lora es False ("no"), y las intensidades no son 0
        if self.cached_lora is not None and not bypass_lora and (strength_model != 0 or strength_clip != 0):
            # Crear copias para no modificar el cache
            model = comfy.sd.load_lora_for_models(self.cached_model, self.cached_clip, self.cached_lora, strength_model, strength_clip)[0]
            clip = comfy.sd.load_lora_for_models(self.cached_model, self.cached_clip, self.cached_lora, strength_model, strength_clip)[1]
        else:
            # Usar cache directamente si bypass_lora es True ("yes") o no hay LoRA
            model = self.cached_model
            clip = self.cached_clip

        return (model, clip, vae)

# Mapeo de nodos
NODE_CLASS_MAPPINGS = {
    "MultiModelLoader": MultiModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiModelLoader": "Multi-Model Loader"
}