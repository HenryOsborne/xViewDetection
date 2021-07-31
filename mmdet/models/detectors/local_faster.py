from ..builder import DETECTORS
from .two_stage_local import TwoStageDetectorLocal


@DETECTORS.register_module
class LocalFasterRCNN(TwoStageDetectorLocal):
    def __init__(self,
                 p_size,
                 batch_size,
                 ori_shape,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(LocalFasterRCNN, self).__init__(
            p_size=p_size,
            batch_size=batch_size,
            ori_shape=ori_shape,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
