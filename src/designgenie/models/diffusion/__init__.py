from controlnet_aux import MLSDdetector, PidiNetDetector, HEDdetector

from .controlnet import StableDiffusionControlNet
from .controlnet_inpaint import StableDiffusionControlNetInpaint

CONTROLNET_MODEL_DICT = {
    "mlsd": {
        "name": "lllyasviel/Annotators",
        "detector": MLSDdetector,
        "model": "lllyasviel/control_v11p_sd15_mlsd",
    },
    "soft_edge": {
        "name": "lllyasviel/Annotators",
        "detector": PidiNetDetector,
        "model": "lllyasviel/control_v11p_sd15_softedge",
    },
    "hed": {
        "name": "lllyasviel/Annotators",
        "detector": HEDdetector,
        "model": "lllyasviel/sd-controlnet-hed",
    },
    "scribble": {
        "name": "lllyasviel/Annotators",
        "detector": HEDdetector,
        "model": "lllyasviel/control_v11p_sd15_scribble",
    },
}

DIFFUSION_MODELS = {
    "controlnet": StableDiffusionControlNet,
    "controlnet_inpaint": StableDiffusionControlNetInpaint,
}


def create_diffusion_model(diffusion_model_name: str, **kwargs):
    assert (
        diffusion_model_name in DIFFUSION_MODELS.keys()
    ), "Diffusion model name must be one of " + ", ".join(DIFFUSION_MODELS.keys())

    return DIFFUSION_MODELS[diffusion_model_name](**kwargs)
