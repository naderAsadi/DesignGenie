from typing import List, Tuple, Dict, Union, Any, Optional
import argparse
from PIL import Image
import wandb


class WandBLogger:
    def __init__(self, config: Dict[str, Any]):
        assert "wandb_project" in config, "Missing `wandb_project` in config"
        self.wandb = wandb.init(
            project=config.wandb_project, name=config.exp_name, config=config
        )

    def log_scalars(self, logs: Dict[str, Union[int, float]]):
        self.wandb.log(logs)

    def log_images(self, logs: Dict[str, List[Image.Image]]):
        wandb.log(
            {
                key: [wandb.Image(image, caption=key) for image in images]
                for key, images in logs.items()
            }
        )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_model", type=str, default="mask2former")
    parser.add_argument("--controlnet_name", type=str, default="hed")
    parser.add_argument(
        "--sd_model", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="/home/nader/DesignGenie/assets/images/",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="/home/nader/DesignGenie/assets/prompts.txt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--n_outputs", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="DesignGenie")
    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="demo")
    args = parser.parse_args()

    return args
