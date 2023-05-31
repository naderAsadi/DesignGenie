from .maskformer import MaskFormer, Mask2Former

SEGMENTATION_MODEL_DICT = {
    "maskformer": MaskFormer,
    "mask2former": Mask2Former,
}
