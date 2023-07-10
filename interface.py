from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import gradio as gr
import numpy as np
from PIL import Image

# import torch

# from ..models import create_diffusion_model, create_segmentation_model
# from ..utils import get_object_mask, visualize_segmentation_map


@dataclass
class AppState:
    """A class to store the memory state of the Gradio App."""

    original_image: Image.Image = None
    predicted_semantic_map = None


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
        # if self.segmentation_model is None:
        #     self.segmentation_model = create_segmentation_model(segmentation_model_name=model_name)

        # predicted_semantic_map = self.segmentation_model.process([image])[0]
        # self._state.predicted_semantic_map = predicted_semantic_map

        # segmentation_map = visualize_segmentation_map(predicted_semantic_map, image)
        # return segmentation_map
        pass

    def _generate_mask(self) -> Image.Image:
        # return get_object_mask(self._state.predicted_semantic_map, class_id=10)
        pass

    def _generate_outputs(
        self,
        image: Image.Image,
        prompt: str,
        object_mask: Image.Image,
        model_name: str,
    ) -> Image.Image:
        # if self.diffusion_model is None:
        #     self.diffusion_model = create_diffusion_model(
        #         diffusion_model_name="controlnet_inpaint", control_model_name=model_name
        #     )

        # outputs = self.diffusion_model.process(
        #     images=[image],
        #     prompts=[prompt],
        #     mask_images=[object_mask],
        #     negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        # )

        # return outputs[0][0]
        pass

    def get_point(self, event: gr.SelectData):
        print(event.index)

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
                    input_image.select(self.get_point)
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

                    # --> Prompt and Num Outputs <--
                    text = gr.Textbox(
                        label="Text prompt(optional)",
                        info="You can describe how the model should redesign the selected segments of your home.",
                    )
                    num_outputs = gr.Slider(
                        value=2,
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
                        output_image1 = gr.Image(
                            interactive=False, label="Output Image 1", type="pil"
                        )
                        output_image2 = gr.Image(
                            interactive=False, label="Output Image 2", type="pil"
                        )

                    with gr.Tab(label="Control Images"):
                        mask_image = gr.Image(
                            interactive=False, label="Generated Mask", type="pil"
                        )
                        masked_image = gr.Image(
                            interactive=False, label="Masked Input Image", type="pil"
                        )
                        control_image = gr.Image(
                            interactive=False, label="Control Image", type="pil"
                        )
                        masked_control_image = gr.Image(
                            interactive=False, label="Masked Control Image", type="pil"
                        )

            # with gr.Row().style(equal_height=True):
            #     # ---> Input and Mask Image holders <---
            #     with gr.Column(scale=1):
            #         input_image = gr.Image(label="Input Image", type="pil")
            #         mask_image = gr.Image(interactive=False, label="Generated Mask", type="pil")

            #     # ---> Segmentation and Mask Generation <---
            #     with gr.Column(scale=2):
            #         segmentation_model = gr.Dropdown(
            #             choices=["mask2former", "maskformer"], label="Segmentation Model"
            #         )
            #         segment_btn = gr.Button(value="Segment Image")
            #         seg_image = gr.Image(interactive=False, label="Segmented Image", type="pil")

            #         segment_btn.click(
            #             self._segment_input,
            #             inputs=[input_image, segmentation_model],
            #             outputs=seg_image,
            #         )

            #         mask_btn = gr.Button(value="Generate Mask")
            #         mask_btn.click(self._generate_mask, outputs=[mask_image])

            # gr.Markdown("<hr>")
            # gr.Markdown(
            #     "<h4 style='text-align: center;'>ControlNet + StableDiffusion Inpaint Pipeline</h4>"
            # )
            # with gr.Row():
            #     with gr.Column():
            #         controlnet_model = gr.Dropdown(
            #             choices=["mlsd", "soft_edge", "hed", "scribble"], label="Controlnet Module"
            #         )
            #         prompt_text = gr.Textbox(label="Prompt", lines=3)
            #         pipeline_btn = gr.Button(value="Run Pipeline")
            #     with gr.Column():
            #         output_image = gr.Image(interactive=False, label="Output Image", type="pil")

            #         pipeline_btn.click(
            #             self._generate_outputs,
            #             inputs=[input_image, prompt_text, mask_image, controlnet_model],
            #             outputs=[output_image]
            #         )

            # with gr.Row():
            #     masked_image = gr.Image(interactive=False, label="Masked Input Image", type="pil")
            #     control_image = gr.Image(interactive=False, label="Control Image", type="pil")
            #     masked_control_image = gr.Image(
            #         interactive=False, label="Masked Control Image", type="pil"
            #     )

        return designgenie_interface


if __name__ == "__main__":
    app = GradioApp()
    app.interface.launch(share=False)
