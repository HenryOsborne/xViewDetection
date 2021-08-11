from ..builder import DETECTORS
from .local_faster import LocalFasterRCNN
import numpy as np
import torch
import random


@DETECTORS.register_module()
class LocalTridentFasterRCNN(LocalFasterRCNN):
    """Implementation of `TridentNet <https://arxiv.org/abs/1901.01892>`_"""

    def __init__(self,
                 p_size,
                 batch_size,
                 ori_shape,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(LocalTridentFasterRCNN, self).__init__(
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
        assert self.backbone.num_branch == self.roi_head.num_branch
        assert self.backbone.test_branch_idx == self.roi_head.test_branch_idx
        self.num_branch = self.backbone.num_branch
        self.test_branch_idx = self.backbone.test_branch_idx

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """make copies of img and gts to fit multi-branch."""
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels, gt_masks)  # 将label切分

        # img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        # img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        # img_metas[0]['scale_factor'] = 1.0
        for i in range(len(trident_img_metas)):
            trident_img_metas[i]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
            trident_img_metas[i]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
            trident_img_metas[i]['scale_factor'] = 1.0

        count_patch = 0
        losses = dict()
        batch_size = self.batch_size
        while count_patch < batch_size:
            try:
                i_patch = random.randint(0, len(coordinates[0]) - 1)
            except:
                continue

            if label_patches[i_patch].shape == torch.Size([0]) or bbox_patches[i_patch] is None \
                    or label_patches[i_patch] is None:
                continue

            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)

            input_bbox = bbox_patches[i_patch]
            input_bbox = tuple([input_bbox] * self.num_branch)

            input_label = label_patches[i_patch]
            input_label = tuple([input_label] * self.num_branch)

            feat_neck = self.extract_feat(input_patch)
            ##############################################################################
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    feat_neck,
                    trident_img_metas,
                    input_bbox,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses = self.update_loss(losses, rpn_losses)
            else:
                proposal_list = proposals
            ##############################################################################
            roi_losses = self.roi_head.forward_train(feat_neck, trident_img_metas, proposal_list,
                                                     input_bbox, input_label,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses = self.update_loss(losses, roi_losses)
            ##############################################################################
            count_patch += 1
        losses = self.loss_mean(losses, batch_size)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):

        assert self.with_bbox, 'Bbox head must be implemented.'

        img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['scale_factor'] = 1.0

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size

        i_patch = 0
        result = []
        return_rpn = False

        for i in range(len(coordinates[0])):
            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)
            feat_neck = self.extract_feat(input_patch)

            if proposals is None:
                num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
                trident_img_metas = img_metas * num_branch
                proposal_list = self.rpn_head.simple_test_rpn(feat_neck, trident_img_metas)
            else:
                proposal_list = proposals

            if return_rpn:
                gl_proposal = self.patch_to_global(proposal_list, i_patch)
                if i_patch > 0:
                    result[0] = np.concatenate([result[0], gl_proposal[0].cpu().numpy()])
                else:
                    result.extend([gl_proposal[0].cpu().numpy()])
                i_patch += 1
                continue

            bbox_results = self.roi_head.simple_test(
                feat_neck, proposal_list, trident_img_metas, rescale=rescale)

            bbox_results = self.patch_to_global(bbox_results[0], i_patch)

            if i_patch > 0:
                result[0] = np.concatenate([result[0], bbox_results[0]])
            else:
                result.extend(bbox_results)
            i_patch += 1

        return result
        #######################################################################
        # """Test without augmentation."""
        # assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(img)
        # if proposals is None:
        #     num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        #     trident_img_metas = img_metas * num_branch
        #     proposal_list = self.rpn_head.simple_test_rpn(x, trident_img_metas)
        # else:
        #     proposal_list = proposals
        #
        # return self.roi_head.simple_test(
        #     x, proposal_list, trident_img_metas, rescale=rescale)
        #######################################################################

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        trident_img_metas = [img_metas * num_branch for img_metas in img_metas]
        proposal_list = self.rpn_head.aug_test_rpn(x, trident_img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

        # return super(LocalTridentFasterRCNN,
        #              self).forward_train(img, trident_img_metas,
        #                                  trident_gt_bboxes, trident_gt_labels)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
