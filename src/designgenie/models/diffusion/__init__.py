from .controlnet import StableDiffusionControlNet
from .controlnet_inpaint import StableDiffusionControlNetInpaint

DIFFUSION_MODELS = {
    "controlnet": StableDiffusionControlNet,
    "controlnet_inpaint": StableDiffusionControlNetInpaint,
}


def create_diffusion_model(diffusion_model_name: str, **kwargs):
    assert (
        diffusion_model_name in DIFFUSION_MODELS.keys()
    ), "Diffusion model name must be one of " + ", ".join(DIFFUSION_MODELS.keys())

    return DIFFUSION_MODELS[diffusion_model_name](**kwargs)
