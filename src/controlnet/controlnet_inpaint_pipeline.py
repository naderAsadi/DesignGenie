from typing import List, Optional, Tuple, Union
import itertools
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from controlnet_aux import MLSDdetector, PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import HfApi
from transformers import pipeline

from .controlnet_pipeline import ControlNetPipeline


class ControlNetInpaintPipeline:
    def __init__(
        self, controlnet_name: str, images_path: str, sd_model: Optional[str] = None
    ):
        self.controlnet_name = controlnet_name
        self.images_path = images_path

        self.pipe = self.create_pipe(controlnet_name=controlnet_name)
        self._images, self._control_images = self.generate_control_images(
            controlnet_name=controlnet_name, images_path=images_path
        )

    def create_pipe(self, controlnet_name: str):
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODELS[controlnet_name]["controlnet"], torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
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
        raise NotImplementedError()
