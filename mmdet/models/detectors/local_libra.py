from ..builder import DETECTORS
from .two_stage_local import TwoStageDetectorLocal


@DETECTORS.register_module
class LocalLibra(TwoStageDetectorLocal):
    def __init__(self,
                 p_size,
                 batch_size,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(LocalLibra, self).__init__(
            p_size=p_size,
            batch_size=batch_size,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
