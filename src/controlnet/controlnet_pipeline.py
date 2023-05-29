from typing import List, Optional, Tuple, Union
import itertools
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from controlnet_aux import MLSDdetector, PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import HfApi
from transformers import pipeline


CONTROLNET_MODELS = {
    "mlsd": {
        "detector_name": "lllyasviel/Annotators",
        "detector_module": MLSDdetector,
        "controlnet": "lllyasviel/control_v11p_sd15_mlsd",
    },
    "soft_edge": {
        "detector_name": "lllyasviel/Annotators",
        "detector_module": PidiNetDetector,  # [PidiNetDetector, HEDdetector]
        "controlnet": "lllyasviel/control_v11p_sd15_softedge",
    },
    "hed": {
        "detector_name": "lllyasviel/Annotators",
        "detector_module": HEDdetector,
        "controlnet": "lllyasviel/sd-controlnet-hed",
    },
}


class ControlNetPipeline:
    def __init__(self, controlnet_name: str, sd_model: str, images_path: str):
        self.controlnet_name = controlnet_name
        self.images_path = images_path

        self.pipe = self.create_pipe(sd_model=sd_model, controlnet_name=controlnet_name)
        self._images, self._control_images = self.generate_control_images(
            controlnet_name=controlnet_name, images_path=images_path
        )

    @property
    def images(self):
        return self._images

    @property
    def control_images(self):
        return self._control_images

    def generate_control_images(
        self, controlnet_name: str, images_path: str
    ) -> List[Image.Image]:
        """Get control images from `images_path` using `controlnet_name` controlnet.

        Args:
            controlnet_name (str): Name of the controlnet processor.
            images_path (str): Path to input Images.

        Returns:
            List[Image.Image]: Processed images.
        """

        processor = CONTROLNET_MODELS[controlnet_name][
            "detector_module"
        ].from_pretrained(CONTROLNET_MODELS[controlnet_name]["detector_name"])
        # Load images in `images_path`
        images = [
            load_image(str(image_path)) for image_path in Path(images_path).glob("*")
        ]
        # Get control images
        control_images = [processor(image) for image in images]

        return images, control_images

    def create_pipe(self, sd_model: str, controlnet_name: str):
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODELS[controlnet_name]["controlnet"], torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=controlnet, torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe

    def generate(
        self,
        prompts: List[str],
        negative_prompt: str,
        n_outputs: Optional[int] = 1,
        num_inference_steps: Optional[int] = 30,
    ):
        "Generate images from `prompts` using `control_images` and `negative_prompt`."

        control_images = self._control_images

        assert len(prompts) == len(
            control_images
        ), "Number of prompts and input images must be equal."

        if n_outputs > 1:
            prompts = list(
                itertools.chain.from_iterable(
                    itertools.repeat(prompt, n_outputs) for prompt in prompts
                )
            )
            control_images = list(
                itertools.chain.from_iterable(
                    itertools.repeat(ctrl_img, n_outputs) for ctrl_img in control_images
                )
            )

        generator = [
            torch.Generator(device="cuda").manual_seed(int(i))
            for i in np.random.randint(50, size=len(prompts))
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
            for idx in range(len(self._control_images))
        ]

        return output_images
