from .maskformer import MaskFormer, Mask2Former

SEGMENTATION_MODEL_DICT = {
    "maskformer": MaskFormer,
    "mask2former": Mask2Former,
}


def create_segmentation_model(segmentation_model_name: str, **kwargs):
    assert (
        segmentation_model_name in SEGMENTATION_MODEL_DICT.keys()
    ), "Segmentation model name must be one of " + ", ".join(
        SEGMENTATION_MODEL_DICT.keys()
    )

    return SEGMENTATION_MODEL_DICT[segmentation_model_name](**kwargs)
