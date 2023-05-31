from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)


class MaskFormer:
    """MaskFormer semantic segmentation model.

    Args:
        model_size (str, optional):
            Size of the MaskFormer model. Defaults to "large".
    """

    def __init__(self, model_size: Optional[str] = "large") -> None:
        assert model_size in [
            "tiny",
            "base",
            "large",
        ], "Model size must be one of 'tiny', 'base', or 'large'"

        self.processor = MaskFormerImageProcessor.from_pretrained(
            f"facebook/maskformer-swin-{model_size}-ade"
        )
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            f"facebook/maskformer-swin-{model_size}-ade"
        )

    def process(self, images: List[Image.Image]):
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to processor for postprocessing
        # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        predicted_semantic_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1] * len(images)]
        )

        return predicted_semantic_maps


class Mask2Former(MaskFormer):
    """Mask2Former semantic segmentation model.

    Args:
        model_size (str, optional):
            Size of the Mask2Former model. Defaults to "large".
    """

    def __init__(self, model_size: Optional[str] = "large") -> None:
        assert model_size in [
            "tiny",
            "base",
            "large",
        ], "Model size must be one of 'tiny', 'base', or 'large'"
        self.processor = AutoImageProcessor.from_pretrained(
            f"facebook/mask2former-swin-{model_size}-ade-semantic"
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            f"facebook/mask2former-swin-{model_size}-ade-semantic"
        )


# class ADESegmentation:
#     def __init__(self, model_name: str):
#         self.processor = MODEL_DICT[model_name]["processor"].from_pretrained(
#             MODEL_DICT[model_name]["name"]
#         )
#         self.model = MODEL_DICT[model_name]["model"].from_pretrained(
#             MODEL_DICT[model_name]["name"]
#         )

#     def predict(self, image: Image.Image):
#         inputs = processor(images=image, return_tensors="pt")
#         outputs = model(**inputs)
#         # model predicts class_queries_logits of shape `(batch_size, num_queries)`
#         # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
#         class_queries_logits = outputs.class_queries_logits
#         masks_queries_logits = outputs.masks_queries_logits

#         # you can pass them to processor for postprocessing
#         # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
#         predicted_semantic_maps = processor.post_process_semantic_segmentation(
#             outputs, target_sizes=[image.size[::-1]]
#         )

#         return predicted_semantic_maps

#     def get_mask(self, predicted_semantic_maps, class_id: int):
#         masks, labels, obj_names = get_masks_from_segmentation_map(
#             predicted_semantic_maps[0]
#         )

#         mask = masks[labels.index(ID)]
#         object_mask = np.logical_not(mask).astype(int)

#         mask = torch.Tensor(mask).repeat(3, 1, 1)
#         object_mask = torch.Tensor(object_mask).repeat(3, 1, 1)

#         return mask, object_mask

#     def get_PIL_mask(self, predicted_semantic_maps, class_id: int):
#         mask, object_mask = self.get_mask(predicted_semantic_maps[0], class_id=class_id)

#         mask = transforms.ToPILImage()(mask)
#         object_mask = transforms.ToPILImage()(object_mask)

#         return mask, object_mask

#     def get_PIL_segmentation_map(self, predicted_semantic_maps):
#         return visualize_segmentation_map(predicted_semantic_maps[0])
