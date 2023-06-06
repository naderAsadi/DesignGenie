from typing import Any, List, Optional, Tuple, Union
import os
from PIL import Image
from random import randint, choices

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from diffusers.utils import load_image


class ImageFolderDataset(Dataset):
    """Dataset class for loading images and prompts from a folder and file path.

    Args:
        images_root (str):
            Path to the folder containing images.
        prompts_path (str):
            Path to the file containing prompts.
        image_size (Tuple[int, int]):
            Size of the images to be loaded.
        extensions (Tuple[str]):
            Tuple of valid image extensions.
    """

    def __init__(
        self,
        images_root: str,
        prompts_path: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        extensions: Tuple[str] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        super().__init__()
        self.image_size = image_size

        self.images_paths, self.prompts = self._make_dataset(
            images_root=images_root, extensions=extensions, prompts_path=prompts_path
        )

        self.to_tensor = transforms.ToTensor()

    def _make_dataset(
        self,
        images_root: str,
        extensions: Tuple[str],
        prompts_path: Optional[str] = None,
    ) -> Tuple[List[str], Union[None, List[str]]]:
        images_paths = []
        for root, _, fnames in sorted(os.walk(images_root)):
            for fname in sorted(fnames):
                if fname.lower().endswith(extensions):
                    images_paths.append(os.path.join(root, fname))

        if prompts_path is not None:
            with open(prompts_path, "r") as f:
                prompts = f.readlines()
        else:
            prompts = None

        return images_paths, prompts

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Union[None, str]]:
        image = load_image(self.images_paths[idx]).resize(self.image_size)
        prompt = self.prompts[idx] if self.prompts is not None else None

        return self.to_tensor(image), prompt
