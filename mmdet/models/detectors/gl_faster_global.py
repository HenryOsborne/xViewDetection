from ..builder import DETECTORS
from .gl_two_stage import GLTwoStage
import torch
import torch.nn as nn


@DETECTORS.register_module
class GLFasterRCNN(GLTwoStage):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(GLFasterRCNN, self).__init__(
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.mode = 1
        self.p_size = (800, 800)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.neck(img, None, None, None, mode=self.mode)
        return x

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(GLTwoStage, self).init_weights(pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights(self.mode)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        if self.mode == 1:
            x = self.neck(img, None, None, None, mode=self.mode)
        else:
            raise ValueError('In global mode,mode should be 1...')

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(x,
                                                                    img_metas,
                                                                    gt_bboxes,
                                                                    gt_labels=None,
                                                                    gt_bboxes_ignore=gt_bboxes_ignore,
                                                                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
