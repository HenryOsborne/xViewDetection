import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os
import mmcv
import cv2
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.models.plugins import DynamicHead


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        # ------------------------------------------------------------------------------------
        if 'use_consistent_supervision' in train_cfg.rcnn:
            self.use_consistent_supervision = train_cfg.rcnn.use_consistent_supervision
        else:
            self.use_consistent_supervision = False

        if 'show_feature' in test_cfg:
            self.show_feature = test_cfg.show_feature
            self.feature_dir = test_cfg.feature_dir
        else:
            self.show_feature = False
            self.feature_dir = None

        self.matched_proposal = []
        if 'assess_proposal_quality' in test_cfg:
            self.assess_proposal_quality = test_cfg.assess_proposal_quality
        else:
            self.assess_proposal_quality = False
        # ------------------------------------------------------------------------------------

        if neck is not None:
            if self.use_consistent_supervision:
                # neck.update({'use_dual_head': True})
                self.neck = build_neck(neck)
            else:
                self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # ------------------------------------------------------------------------------------
        x = self.backbone(img)
        if self.with_neck:
            if self.use_consistent_supervision:
                x, y = self.neck(x)
                return x, y
            else:
                x = self.neck(x)
                return x
        else:
            return x
        # ------------------------------------------------------------------------------------

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # ------------------------------------------------------------------------------------
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        # ------------------------------------------------------------------------------------

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ------------------------------------------------------------------------------------
        if self.use_consistent_supervision:
            roi_losses = self.roi_head.forward_train(x, y, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        else:
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        # ------------------------------------------------------------------------------------

        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        # ------------------------------------------------------------------------------------
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        if self.show_feature:
            # from PIL import Image
            # import numpy as np
            #
            # feature = x[0][0][0].cpu()
            # feature = 1.0 / (1 + np.exp(-1 * feature))
            # feature = np.round(feature * 255)
            # img = transforms.ToPILImage()(feature).convert('RGB')
            # img = img.resize((800, 800), Image.ANTIALIAS)
            # img_name = img_metas[0]['ori_filename'].split('.')[0]
            # os.makedirs('feature', exist_ok=True)
            # img.save(self.feature_dir + str(img_name) + '_p2.jpg')
            # --------------------------------------------------------------------------------
            import numpy as np
            from mmdet.utils import featuremap_to_heatmap
            img_name = img_metas[0]['ori_filename'].split('.')[0]
            heatmap = featuremap_to_heatmap(x[0])
            image = mmcv.imread(img_metas[0]['filename'])
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
            superimposed_img = heatmap * 0.4 + image  # 这里的0.4是热力图强度因子
            cv2.imwrite(os.path.join(self.feature_dir, str(img_name) + '_p2.jpg'), superimposed_img)  # 将图像保存到硬盘
        # ------------------------------------------------------------------------------------

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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
