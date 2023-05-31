from controlnet_aux import MLSDdetector, PidiNetDetector, HEDdetector

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


from .controlnet import StableDiffusionControlNet
from .controlnet_inpaint import StableDiffusionControlNetInpaint
