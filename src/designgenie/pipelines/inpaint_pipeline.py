from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ..data import ImageFolderDataset
from ..models import create_diffusion_model, create_segmentation_model
from ..utils import get_masked_images


class InpaintPipeline:
    def __init__(
        self,
        segmentation_model_name: str,
        diffusion_model_name: str,
        control_model_name: str,
        images_root: str,
        prompts_path: Optional[str] = None,
        sd_model_name: Optional[str] = "runwayml/stable-diffusion-inpainting",
        image_size: Optional[Tuple[int, int]] = (512, 512),
        image_extensions: Optional[Tuple[str]] = (".jpg", ".jpeg", ".png", ".webp"),
        segmentation_model_size: Optional[str] = "large",
    ):
        self.segmentation_model = create_segmentation_model(
            segmentation_model_name=segmentation_model_name,
            model_size=segmentation_model_size,
        )

        self.diffusion_model = create_diffusion_model(
            diffusion_model_name=diffusion_model_name,
            control_model_name=control_model_name,
            sd_model_name=sd_model_name,
        )

        self.data_loader = self.build_data_loader(
            images_root=images_root,
            prompts_path=prompts_path,
            image_size=image_size,
            image_extensions=image_extensions,
        )

    def build_data_loader(
        self,
        images_root: str,
        prompts_path: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        image_extensions: Optional[Tuple[str]] = (".jpg", ".jpeg", ".png", ".webp"),
        batch_size: Optional[int] = 1,
    ) -> DataLoader:
        dataset = ImageFolderDataset(
            images_root, prompts_path, image_size, image_extensions
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )

        return data_loader

    def run(self, data_loader: Optional[DataLoader] = None) -> List[Dict[str, Any]]:
        if data_loader is not None:
            self.data_loader = data_loader

        results = []
        for idx, (images, prompts) in enumerate(self.data_loader):
            images = [to_pil_image(img) for img in images]

            semantic_maps = self.segmentation_model.process(images)

            object_masks = [
                get_object_mask(seg_map, class_id=0) for seg_map in semantic_maps
            ]

            outputs = self.diffusion_model.process(
                images=images,
                prompts=[prompts[0]],
                mask_images=object_masks,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            )
            results += outputs["output_images"]

        return results
