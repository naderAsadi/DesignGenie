from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader

from ..data import ImageFolderDataset
from ..models import create_diffusion_model, create_segmentation_model


class InpaintPipeline:
    def __init__(
        self,
        segmentation_model_name: str,
        control_model_name: str,
        sd_model_name: str,
        images_root: str,
        prompts_path: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        image_extensions: Optional[Tuple[str]] = (".jpg", ".jpeg", ".png", ".webp"),
        segmentation_model_size: Optional[str] = "large",
    ):
        self.segmentation_model_name = segmentation_model_name

        self.data_loader = self._build_data_loader(
            images_root=images_root,
            prompts_path=prompts_path,
            image_size=image_size,
            image_extensions=image_extensions,
        )

    def _build_data_loader(
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
