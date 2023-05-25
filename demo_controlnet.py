from typing import List, Optional, Tuple, Union
import math
from PIL import Image
import wandb

from src import ControlNetPipeline, WandBLogger, parser


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main(args):
    # Load prompts from `prompts_path`
    with open(args.prompts_path, "r") as f:
        prompts = f.readlines()

    # Load controlnet
    controlnet = ControlNetPipeline(
        controlnet_name=args.controlnet_name,
        sd_model=args.sd_model,
        images_path=args.images_path,
    )

    output = controlnet.generate(
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        n_outputs=args.n_outputs,
        num_inference_steps=args.num_inference_steps,
    )

    output_images = [
        image_grid(imgs=output[idx], rows=1, cols=args.n_outputs)
        for idx in range(len(output))
    ]

    logger = WandBLogger(config=args)
    logger.log_images(
        logs={
            "generated_images": output_images,
            "original_images": controlnet.images,
            "control_images": controlnet.control_images,
        }
    )


if __name__ == "__main__":
    args = parser()
    main(args)
