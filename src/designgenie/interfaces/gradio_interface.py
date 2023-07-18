from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

from ..models import create_diffusion_model, create_segmentation_model
from ..utils import (
    get_masked_images,
    visualize_segmentation_map,
    get_masks_from_segmentation_map,
)


# points color and marker
COLOR = (255, 0, 0)


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
        self.state = AppState()

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
        self.state.predicted_semantic_map = predicted_semantic_map

        segmentation_map = visualize_segmentation_map(predicted_semantic_map, image)
        return segmentation_map

    def _generate_outputs(
        self,
        prompt: str,
        model_name: str,
        n_outputs: int,
        inference_steps: int,
        strength: float,
        guidance_scale: float,
        eta: float,
    ) -> Image.Image:
        if self.diffusion_model is None:
            self.diffusion_model = create_diffusion_model(
                diffusion_model_name="controlnet_inpaint", control_model_name=model_name
            )

        control_image = self.diffusion_model.generate_control_images(
            images=[self.state.original_image]
        )[0]

        image_mask, masked_control_image = get_masked_images(
            control_image,
            self.state.predicted_semantic_map,
            self.state.input_coordinates,
        )

        outputs = self.diffusion_model.process(
            images=[self.state.original_image],
            prompts=[prompt],
            mask_images=[image_mask],
            control_images=[masked_control_image],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            n_outputs=n_outputs,
        )

        return (
            *outputs["output_images"][0],
            control_image,
            image_mask,
        )

    def image_change(self, input_image):
        input_image = input_image.resize((768, 512))
        self.state.original_image = input_image
        return input_image

    def clear_coordinates(self):
        self.state.input_coordinates = []

    def get_coordinates(self, event: gr.SelectData, input_image: Image.Image):
        w, h = tuple(event.index)
        self.state.input_coordinates.append((h, w))
        print(self.state.input_coordinates)

        return Image.fromarray(
            cv2.drawMarker(
                np.asarray(input_image), event.index, COLOR, markerSize=20, thickness=5
            )
        )

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
                with gr.Column():
                    gr.Markdown("### Stable Diffusion Parameters")
                    with gr.Row():
                        with gr.Column():
                            inference_steps = gr.Number(
                                value=30, label="Number of inference steps."
                            )
                            strength = gr.Number(value=1.0, label="Strength.")
                        with gr.Column():
                            guidance_scale = gr.Number(value=7.5, label="Guidance scale.")
                            eta = gr.Number(value=0.0, label="Eta.")

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    # --> Input Image and Segmentation <--
                    input_image = gr.Image(label="Input Image", type="pil")
                    input_image.select(
                        self.get_coordinates,
                        inputs=[input_image],
                        outputs=[input_image],
                    )
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
                        control_labels = ["Control Image", "Generated Mask"]
                        control_images = [
                            gr.Image(interactive=False, label=label, type="pil")
                            for label in control_labels
                        ]

                submit_btn.click(
                    self._generate_outputs,
                    inputs=[
                        text,
                        controlnet_model,
                        num_outputs,
                        inference_steps,
                        strength,
                        guidance_scale,
                        eta,
                    ],
                    outputs=output_images + control_images,
                )

        return designgenie_interface
