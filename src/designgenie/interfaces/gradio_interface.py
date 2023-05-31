from typing import Any, Dict, List, Optional, Tuple, Union
import gradio as gr
import numpy as np
from PIL import Image

from src import ADESegmentationModel


class DesignGenie:
    def __init__(self):
        self._interface = self.build_interface()
        self.segmentation_model = None
        self.controlnet_pipe = None

    @property
    def interface(self):
        return self._interface

    def segment_input(image: Image.Image, model_name: str):
        """Segment the input image using the given model."""
        if self.segmentation_model is None:
            self.segmentation_model = ADESegmentationModel(model_name=model_name)

        predicted_semantic_maps = self.segmentation_model.get_mask_PIL(
            image=image, image_id=10
        )

        return predicted_semantic_maps

    def build_interface():
        """Builds the Gradio interface for the DesignGenie app."""
        with gr.Blocks() as designgenie_interface:
            gr.Markdown("<h1 style='text-align: center;'>DesignGenie</h1>")
            gr.Markdown(
                "<p style='text-align: center;'>A tool to generate designs from a given image.</p>"
            )

            with gr.Row().style(equal_height=True):
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image")
                    mask_image = gr.Image(interactive=False, label="Generated Mask")

                with gr.Column(scale=2):
                    segmentation_model = gr.Dropdown(
                        choices=["Mask2Former", "OneFormer"], label="Segmentation Model"
                    )
                    segment_btn = gr.Button(value="Segment Image")
                    seg_image = gr.Image(interactive=False, label="Segmented Image")

                    segment_btn.click(
                        segment_input,
                        inputs=[input_image, segmentation_model],
                        outputs=seg_image,
                    )

                    mask_btn = gr.Button(value="Generate Mask")
                    # mask_btn.click(func, inputs=[], outputs=[])

            gr.Markdown("<hr>")
            gr.Markdown(
                "<h4 style='text-align: center;'>ControlNet + StableDiffusion Inpaint Pipeline</h4>"
            )
            with gr.Row():
                with gr.Column():
                    controlnet_model = gr.Dropdown(
                        choices=["MLSD", "Soft Edge", "HED"], label="Controlnet Module"
                    )
                with gr.Column():
                    controlnet_btn = gr.Button(value="Generate Control Image")
                    pipeline_btn = gr.Button(value="Run Pipeline")
            with gr.Row():
                masked_image = gr.Image(interactive=False, label="Masked Input Image")
                control_image = gr.Image(interactive=False, label="Control Image")
                masked_control_image = gr.Image(
                    interactive=False, label="Masked Control Image"
                )
            output_image = gr.Image(interactive=False, label="Output Image")

        return designgenie_interface


app = DesignGenie()
app.interface.launch()
