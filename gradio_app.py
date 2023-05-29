from typing import Any, Dict, List, Optional, Tuple, Union
import gradio as gr
from PIL import Image


def inference(image: Image.Image, segmentation_model: str):
    pass


def create_gradio_interface(
    title: str,
    description: str,
    function: callable,
    examples: Optional[List[str]] = None,
):
    demo = gr.Interface(
        fn=inference,
        inputs=[
            gr.inputs.Image(type="pil"),
            gr.inputs.Radio(
                ["Mask2Former", "OneFormer"],
                type="value",
                default="Mask2Former",
                label="Segmentation Model",
            ),
        ],
        outputs=gr.outputs.Image(type="pil"),
        title=title,
        description=description,
        examples=examples,
    )

    return demo


if __name__ == "__main__":
    title = "DesignGenie"
    description = "Gradio Demo for DesignGenie. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please use a cropped portrait picture for best results similar to the examples below."
    # examples = [
    #     ["groot.jpeg", "version 2 (🔺 robustness,🔻 stylization)"],
    #     ["gongyoo.jpeg", "version 1 (🔺 stylization, 🔻 robustness)"],
    # ]

    demo = create_gradio_interface(title, description, function=inference)
    demo.launch()
