from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

from ..models import create_diffusion_model, create_segmentation_model
from ..utils import (
    get_object_mask,
    visualize_segmentation_map,
    get_masks_from_segmentation_map,
)


# points color and marker
COLORS = [(255, 0, 0), (0, 255, 0)]
MARKERS = [1, 5]

@dataclass
class AppState:
    """A class to store the memory state of the Gradio App."""

    original_image: Image.Image = None
    predicted_semantic_map: torch.Tensor = None
    input_coordinates: List[int] = field(default_factory=list)
    n_outputs: int = 2


class GradioApp:
    def __init__(self):
        self._interface = self.build_interface()
        self._state = AppState()

        self.segmentation_model = None
        self.diffusion_model = None

    @property
    def interface(self):
        return self._interface

    def _segment_input(self, image: Image.Image, model_name: str) -> Image.Image:
        """Segment the input image using the given model."""
        if self.segmentation_model is None:
            self.segmentation_model = create_segmentation_model(
                segmentation_model_name=model_name
            )

        predicted_semantic_map = self.segmentation_model.process([image])[0]
        self._state.predicted_semantic_map = predicted_semantic_map

        segmentation_map = visualize_segmentation_map(predicted_semantic_map, image)
        return segmentation_map

    def _generate_outputs(
        self, prompt: str, model_name: str, n_outputs: int
    ) -> Image.Image:
        if self.diffusion_model is None:
            self.diffusion_model = create_diffusion_model(
                diffusion_model_name="controlnet_inpaint", control_model_name=model_name
            )

        object_mask = get_object_mask(
            self._state.predicted_semantic_map, self._state.input_coordinates
        )

        outputs = self.diffusion_model.process(
            images=[self._state.original_image],
            prompts=[prompt],
            mask_images=[object_mask],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            n_outputs=n_outputs,
        )

        return (
            *outputs["output_images"][0],
            outputs["control_images"][0],
            outputs["mask_images"][0],
        )

    def image_change(self, input_image):
        input_image = input_image.resize((768, 512))
        self._state.original_image = input_image
        return input_image

    def clear_coordinates(self):
        self._state.input_coordinates = []

    def get_coordinates(self, event: gr.SelectData):
        w, h = tuple(event.index)
        self._state.input_coordinates.append((h, w))
        print(self._state.input_coordinates)

    def build_interface(self):
        """Builds the Gradio interface for the DesignGenie app."""
        with gr.Blocks() as designgenie_interface:
            # --> App Header <---
            with gr.Row():
                # --> Description <--
                with gr.Column():
                    gr.Markdown(
                        """
                        # DesignGenie

                        An AI copilot for home interior design. It identifies various sections of your home and generates personalized designs for the selected sections using ContolNet and StableDiffusion.
                        """
                    )
                # --> Model Selection <--
                with gr.Column():
                    with gr.Row():
                        segmentation_model = gr.Dropdown(
                            choices=["mask2former", "maskformer"],
                            label="Segmentation Model",
                            value="mask2former",
                            interactive=True,
                        )
                        controlnet_model = gr.Dropdown(
                            choices=["mlsd", "soft_edge", "hed", "scribble"],
                            label="Controlnet Module",
                            value="mlsd",
                            interactive=True,
                        )

            # --> Model Parameters <--
            with gr.Accordion(label="Parameters", open=False):
                with gr.Row():
                    pass

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    # --> Input Image and Segmentation <--
                    input_image = gr.Image(label="Input Image", type="pil")
                    input_image.select(self.get_coordinates)
                    input_image.upload(
                        self.image_change, inputs=[input_image], outputs=[input_image]
                    )

                    with gr.Row():
                        gr.Markdown(
                            """
                            1. Select your input image.
                            2. Click on `Segment Image` button.
                            3. Choose the segments that you want to redisgn by simply clicking on the image.
                            """
                        )
                        with gr.Column():
                            segment_btn = gr.Button(
                                value="Segment Image", variant="primary"
                            )
                            clear_btn = gr.Button(value="Clear")

                            segment_btn.click(
                                self._segment_input,
                                inputs=[input_image, segmentation_model],
                                outputs=input_image,
                            )
                            clear_btn.click(self.clear_coordinates)

                    # --> Prompt and Num Outputs <--
                    text = gr.Textbox(
                        label="Text prompt(optional)",
                        info="You can describe how the model should redesign the selected segments of your home.",
                    )
                    num_outputs = gr.Slider(
                        value=3,
                        minimum=1,
                        maximum=5,
                        step=1,
                        interactive=True,
                        label="Number of Generated Outputs",
                        info="Number of design outputs you want the model to generate.",
                    )

                    submit_btn = gr.Button(value="Submit", variant="primary")

                with gr.Column():
                    with gr.Tab(label="Output Images"):
                        output_images = [
                            gr.Image(
                                interactive=False, label=f"Output Image {i}", type="pil"
                            )
                            for i in range(3)
                        ]

                    with gr.Tab(label="Control Images"):
                        control_labels = ["Generated Mask", "Control Image"]
                        control_images = [
                            gr.Image(interactive=False, label=label, type="pil")
                            for label in control_labels
                        ]

                submit_btn.click(
                    self._generate_outputs,
                    inputs=[text, controlnet_model, num_outputs],
                    outputs=output_images + control_images,
                )

        return designgenie_interface
