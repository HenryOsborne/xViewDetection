from ..builder import DETECTORS
from .gl_two_stage import GLTwoStage
import torch
import torch.nn as nn
import cv2
import numpy as np
import mmcv


@DETECTORS.register_module
class GlobalGLGA(GLTwoStage):
    def __init__(self,
                 p_size,
                 batch_size,
                 mode,
                 ori_shape,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(GlobalGLGA, self).__init__(
            mode=mode,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.batch_size = batch_size
        self.p_size = p_size
        self.ori_shape = ori_shape

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.neck(img, None, None, None, mode=self.MODE)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        if self.MODE == 1:
            x = self.neck(img, None, None, None, mode=self.MODE)

        elif self.MODE == 3:
            input_img = cv2.imread(img_metas[0]['filename'])
            input_img = input_img.astype(np.float32)
            input_img, scale_factor = self.img_resize(input_img, self.ori_shape)
            patches, coordinates, templates, sizes, ratios = \
                self.global_to_patch(input_img, self.p_size)
            x = self.neck(img, patches, coordinates, ratios, templates, mode=self.MODE)
        else:
            raise ValueError('In global mode,mode should be 1 or 3 ...')

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

        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        import numpy as np

        if self.MODE == 1:
            x = self.neck(img, None, None, None, mode=self.MODE)
        elif self.MODE == 3:
            input_img = cv2.imread(img_metas[0]['filename'])
            input_img = input_img.astype(np.float32)
            input_img, scale_factor = self.img_resize(input_img, self.ori_shape)

            patches, coordinates, templates, sizes, ratios = self.global_to_patch(input_img, self.p_size)
            x = self.neck(img, patches, coordinates, ratios, templates, mode=self.MODE)

        else:
            raise ValueError('wrong mode:{}'.format(self.MODE))

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        # ------------------------------------------------------------------------------------
        if self.assess_proposal_quality:
            from mmdet.core.bbox.iou_calculators import build_iou_calculator
            import numpy as np

            gt_bboxes = kwargs['gt_bboxes'][0][0]
            bboxes = proposal_list[0]
            iou_calculator = dict(type='BboxOverlaps2D')
            iou_calculator = build_iou_calculator(iou_calculator)
            if len(gt_bboxes) != 0:
                overlaps = iou_calculator(gt_bboxes, bboxes)
                max_overlaps, _ = overlaps.max(dim=0)
                max_overlaps = max_overlaps.cpu().numpy()
                idx = max_overlaps >= 0.5
                max_overlaps = max_overlaps[idx].tolist()
                self.matched_proposal.extend(max_overlaps)
        # ------------------------------------------------------------------------------------

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def img_resize(self, img, size):
        # img = (img[0].cpu()).numpy()
        # img = img.transpose(1, 2, 0)
        img, w_scale, h_scale = mmcv.imresize(
            img, size, return_scale=True)
        # img = img.transpose(2, 0, 1)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = mmcv.imnormalize(img, mean, std, True)
        img = img.transpose(2, 0, 1)  # img = bgr2rgb(img)
        img = torch.from_numpy(img).cuda()
        img = img.unsqueeze(0)
        return img, scale_factor
