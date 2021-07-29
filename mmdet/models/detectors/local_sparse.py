from ..builder import DETECTORS
from .two_stage_local import TwoStageDetectorLocal


@DETECTORS.register_module()
class LocalSparseRCNN(TwoStageDetectorLocal):
    r"""Implementation of `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self, *args, **kwargs):
        def __init__(self,
                     backbone,
                     neck=None,
                     rpn_head=None,
                     roi_head=None,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None):
            super(LocalSparseRCNN, self).__init__(
                backbone=backbone,
                neck=neck,
                rpn_head=rpn_head,
                roi_head=roi_head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=pretrained)
