from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

from .controlnet import StableDiffusionControlNet, MODEL_DICT


class StableDiffusionControlNetInpaint(StableDiffusionControlNet):
    """StableDiffusion with ControlNet model for inpainting images based on prompts.

    Args:
        control_model_name (str):
            Name of the controlnet processor.
        sd_model_name (str):
            Name of the StableDiffusion model.
    """

    def __init__(
        self,
        control_model_name: str,
        sd_model_name: Optional[str] = "runwayml/stable-diffusion-inpainting",
    ) -> None:
        super().__init__(
            control_model_name=control_model_name,
            sd_model_name=sd_model_name,
        )

    def create_pipe(
        self, sd_model_name: str, control_model_name: str
    ) -> StableDiffusionControlNetInpaintPipeline:
        """Create a StableDiffusionControlNetInpaintPipeline.

        Args:
            sd_model_name (str): StableDiffusion model name.
            control_model_name (str): Name of the ControlNet module.

        Returns:
            StableDiffusionControlNetInpaintPipeline
        """
        controlnet = ControlNetModel.from_pretrained(
            MODEL_DICT[control_model_name]["model"], torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            sd_model_name, controlnet=controlnet, torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe

    def process(
        self,
        images: List[Image.Image],
        prompts: List[str],
        mask_images: List[Image.Image],
        control_images: Optional[List[Image.Image]] = None,
        negative_prompt: Optional[str] = None,
        n_outputs: Optional[int] = 1,
        num_inference_steps: Optional[int] = 30,
        strength: Optional[float] = 1.0,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
    ) -> List[List[Image.Image]]:
        """Inpaint images based on `prompts` using `control_images` and `mask_images`.

        Args:
            images (List[Image.Image]): Input images.
            prompts (List[str]): List of prompts.
            mask_images (List[Image.Image]): List of mask images.
            control_images (Optional[List[Image.Image]], optional): List of control images. Defaults to None.
            negative_prompt (Optional[str], optional): Negative prompt. Defaults to None.
            n_outputs (Optional[int], optional): Number of generated outputs. Defaults to 1.
            num_inference_steps (Optional[int], optional): Number of inference iterations. Defaults to 30.

        Returns:
            List[List[Image.Image]]
        """

        if control_images is None:
            control_images = self.generate_control_images(images)

        assert len(prompts) == len(
            control_images
        ), "Number of prompts and input images must be equal."

        if n_outputs > 1:
            prompts = self._repeat(prompts, n=n_outputs)
            images = self._repeat(images, n=n_outputs)
            control_images = self._repeat(control_images, n=n_outputs)
            mask_images = self._repeat(mask_images, n=n_outputs)

        generator = [
            torch.Generator(device="cuda").manual_seed(int(i))
            for i in np.random.randint(max(len(prompts), 16), size=len(prompts))
        ]

        output = self.pipe(
            prompts,
            image=images,
            control_image=control_images,
            mask_image=mask_images,
            negative_prompt=[negative_prompt] * len(prompts),
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        output_images = [
            output.images[idx * n_outputs : (idx + 1) * n_outputs]
            for idx in range(len(images) // n_outputs)
        ]

        return {
            "output_images": output_images,
            "control_images": control_images,
            "mask_images": mask_images,
        }
