from typing import List, Optional, Tuple, Union
import itertools
from PIL import Image
import numpy as np
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from . import CONTROLNET_MODEL_DICT as MODEL_DICT


class StableDiffusionControlNet:
    """ControlNet pipeline for generating images from prompts.

    Args:
        control_model_name (str):
            Name of the controlnet processor.
        sd_model_name (str):
            Name of the StableDiffusion model.
    """

    def __init__(
        self,
        control_model_name: str,
        sd_model_name: Optional[str] = "runwayml/stable-diffusion-v1-5",
    ) -> None:
        self.processor = MODEL_DICT[control_model_name]["detector"].from_pretrained(
            MODEL_DICT[control_model_name]["name"]
        )
        self.pipe = self.create_pipe(
            sd_model_name=sd_model_name, controlnet_name=control_model_name
        )

    def _repeat(self, items: List[Any], n: int) -> List[Any]:
        """Repeat items in a list n times.

        Args:
            items (List[Any]): List of items to be repeated.
            n (int): Number of repetitions.

        Returns:
            List[Any]: List of repeated items.
        """
        return list(
            itertools.chain.from_iterable(itertools.repeat(item, n) for item in items)
        )

    def generate_control_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Generate control images from input images.

        Args:
            images (List[Image.Image]): Input images.

        Returns:
            List[Image.Image]: Control images.
        """
        return [self.processor(image) for image in images]

    def create_pipe(
        self, sd_model_name: str, control_model_name: str
    ) -> StableDiffusionControlNetPipeline:
        """Create a StableDiffusionControlNetPipeline.

        Args:
            sd_model_name (str): StableDiffusion model name.
            control_model_name (str): Name of the ControlNet module.

        Returns:
            StableDiffusionControlNetPipeline
        """
        controlnet = ControlNetModel.from_pretrained(
            MODEL_DICT[control_model_name]["model"], torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
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
        negative_prompt: Optional[str] = None,
        n_outputs: Optional[int] = 1,
        num_inference_steps: Optional[int] = 30,
    ) -> List[List[Image.Image]]:
        """Generate images from `prompts` using `control_images` and `negative_prompt`.

        Args:
            images (List[Image.Image]): Input images.
            prompts (List[str]): List of prompts.
            negative_prompt (Optional[str], optional): Negative prompt. Defaults to None.
            n_outputs (Optional[int], optional): Number of generated outputs. Defaults to 1.
            num_inference_steps (Optional[int], optional): Number of inference iterations. Defaults to 30.

        Returns:
            List[List[Image.Image]]
        """

        control_images = self.generate_control_images(images)

        assert len(prompts) == len(
            control_images
        ), "Number of prompts and input images must be equal."

        if n_outputs > 1:
            prompts = self._repeat(prompts, n=n_outputs)
            control_images = self._repeat(control_images, n=n_outputs)

        generator = [
            torch.Generator(device="cuda").manual_seed(int(i))
            for i in np.random.randint(len(prompts), size=len(prompts))
        ]

        output = self.pipe(
            prompts,
            image=control_images,
            negative_prompt=[negative_prompt] * len(prompts),
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        output_images = [
            output.images[idx * n_outputs : (idx + 1) * n_outputs]
            for idx in range(len(images))
        ]

        return output_images
