class MultiLatentSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "selector": (["latent_1", "latent_2", "latent_3", "latent_4"], {"default": "latent_1"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "select_latent"
    CATEGORY = "utilities/latent"
    DISPLAY_NAME = "Multi-Latent Selector"

    def select_latent(self, latent_1=None, latent_2=None, latent_3=None, latent_4=None, selector="latent_1"):
        if selector == "latent_1":
            if latent_1 is None:
                raise ValueError("latent_1 is not connected but was selected in the selector")
            return (latent_1,)
        elif selector == "latent_2":
            if latent_2 is None:
                raise ValueError("latent_2 is not connected but was selected in the selector")
            return (latent_2,)
        elif selector == "latent_3":
            if latent_3 is None:
                raise ValueError("latent_3 is not connected but was selected in the selector")
            return (latent_3,)
        elif selector == "latent_4":
            if latent_4 is None:
                raise ValueError("latent_4 is not connected but was selected in the selector")
            return (latent_4,)
        else:
            raise ValueError(f"Invalid selector value: {selector}")

NODE_CLASS_MAPPINGS = {
    "MultiLatentSelector": MultiLatentSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLatentSelector": "Multi-Latent Selector"
}