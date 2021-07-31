from ..builder import DETECTORS
from .two_stage_local import TwoStageDetectorLocal
import random
import torch
import numpy as np


@DETECTORS.register_module()
class LocalSparseRCNN(TwoStageDetectorLocal):
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
        super(LocalSparseRCNN, self).__init__(
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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None, 'Sparse R-CNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'Sparse R-CNN does not instance segmentation'

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels, gt_masks)

        img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['scale_factor'] = 1.0

        count_patch = 0
        losses = dict()
        batch_size = 4
        while count_patch < batch_size:
            try:
                i_patch = random.randint(0, len(coordinates[0]) - 1)
            except:
                continue

            if label_patches[i_patch].shape == torch.Size([0]) or bbox_patches[i_patch] is None or label_patches[
                i_patch] is None:
                continue

            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)

            input_bbox = bbox_patches[i_patch]
            input_bbox = input_bbox.unsqueeze(0)

            input_label = label_patches[i_patch]
            input_label = input_label.unsqueeze(0)

            feat_neck = self.extract_feat(input_patch)

            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.forward_train(feat_neck, img_metas)

            roi_losses = self.roi_head.forward_train(
                feat_neck,
                proposal_boxes,
                proposal_features,
                img_metas,
                input_bbox,
                input_label,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
                imgs_whwh=imgs_whwh)

            losses = self.update_loss(losses, roi_losses)
            count_patch += 1
        losses = self.loss_mean(losses, batch_size)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        assert self.with_bbox, "Bbox head must be implemented."
        img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['scale_factor'] = 1.0

        patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)

        i_patch = 0
        result = []
        return_rpn = False

        for i in range(len(coordinates[0])):
            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)
            feat_neck = self.extract_feat(input_patch)

            # RPN
            proposal_boxes, proposal_features, imgs_whwh = \
                self.rpn_head.simple_test_rpn(feat_neck, img_metas)

            #########################################################################
            if return_rpn:
                gl_proposal = self.patch_to_global(proposal_boxes, i_patch)
                if i_patch > 0:
                    result[0] = np.concatenate([result[0], gl_proposal[0].cpu().numpy()])
                else:
                    result.extend([gl_proposal[0].cpu().numpy()])
                i_patch += 1
                continue
            #########################################################################

            # ROI
            bbox_results = self.roi_head.simple_test(feat_neck,
                                                     proposal_boxes,
                                                     proposal_features,
                                                     img_metas,
                                                     imgs_whwh=imgs_whwh,
                                                     rescale=rescale)

            bbox_results = self.patch_to_global(bbox_results[0], i_patch)

            if i_patch > 0:
                result[0] = np.concatenate([result[0], bbox_results[0]])
            else:
                result.extend(bbox_results)
            i_patch += 1

        return result

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(self.p_size[0], self.p_size[1], 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs
