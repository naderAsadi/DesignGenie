from typing import Any, List, Optional, Tuple, Union
from functools import reduce
import numpy as np
from PIL import Image
import requests
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


def visualize_segmentation_map(
    semantic_map: torch.Tensor, original_image: Image.Image
) -> Image.Image:
    """
    Visualizes a segmentation map by overlaying it on the original image.

    Args:
        semantic_map (torch.Tensor): Segmentation map tensor.
        original_image (Image.Image): Original image.

    Returns:
        Image.Image: Overlay image with segmentation map.
    """
    # Convert to RGB
    color_seg = np.zeros(
        (semantic_map.shape[0], semantic_map.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[semantic_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(original_image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    return Image.fromarray(img)


def get_masks_from_segmentation_map(
    semantic_map: torch.Tensor,
) -> Tuple[List[np.array], List[int], List[str]]:
    """
    Extracts masks, labels, and object names from a segmentation map.

    Args:
        semantic_map (torch.Tensor): Segmentation map tensor.

    Returns:
        Tuple[List[np.array], List[int], List[str]]: Tuple containing masks, labels, and object names.
    """
    masks = []
    labels = []
    obj_names = []
    for label, color in enumerate(np.array(ade_palette())):
        mask = np.ones(
            (semantic_map.shape[0], semantic_map.shape[1]), dtype=np.uint8
        )  # height, width
        indices = semantic_map == label
        mask[indices] = 0

        if indices.sum() > 0:
            masks.append(mask)
            labels.append(label)
            obj_names.append(ADE_LABELS[str(label)])

    return masks, labels, obj_names


def get_mask_from_coordinates(
    segmentation_maps: List[np.array], coordinates: Tuple[int, int]
):
    """
    Retrieves a mask from a list of segmentation maps based on given coordinates.

    Args:
        segmentation_maps (List[np.array]): List of segmentation maps.
        coordinates (Tuple[int, int]): Coordinates to filter the masks.

    Returns:
        np.array: Combined mask from the segmentation maps.
    """
    masks = []
    for seg_map in segmentation_maps:
        for coordinate in coordinates:
            if seg_map[coordinate] == 0:
                masks.append(seg_map)

    return reduce(np.multiply, masks)


def get_masked_images(
    control_image: Image.Image,
    semantic_map: torch.Tensor,
    coordinates: List[Tuple[int, int]],
    return_tensors: bool = False,
) -> Union[torch.Tensor, Image.Image]:
    """
    Retrieves masked images based on given control image, segmentation map, and coordinates.

    Args:
        control_image (Image.Image): Control image.
        semantic_map (torch.Tensor): Segmentation map tensor.
        coordinates (List[Tuple[int, int]]): List of coordinates.
        return_tensors (bool, optional): Whether to return masked images as tensors. Defaults to False.

    Returns:
        Union[torch.Tensor, Image.Image]: Masked image tensor or PIL image.
    """
    masks, labels, obj_names = get_masks_from_segmentation_map(semantic_map)

    mask = get_mask_from_coordinates(masks, coordinates)

    mask_image = np.logical_not(mask).astype(int)
    mask_image = torch.Tensor(mask_image).repeat(3, 1, 1)

    mask = torch.Tensor(mask).repeat(3, 1, 1)
    control_image = transforms.ToTensor()(control_image)
    masked_control_image = transforms.ToPILImage()(mask * control_image)

    if not return_tensors:
        mask_image = to_pil_image(mask_image)

    return mask_image, masked_control_image


ADE_LABELS = requests.get(
    "https://huggingface.co/datasets/huggingface/label-files/raw/main/ade20k-id2label.json"
).json()


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
