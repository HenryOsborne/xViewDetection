from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import numpy as np
from torch.autograd import Variable
import random


@DETECTORS.register_module()
class Dynamic_Local(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(Dynamic_Local, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.p_size = (800, 800)

    def get_patch_info(self, shape, p_size):
        """
        shape: origin image size, (x, y)
        p_size: patch size (square)
        return: n_x, n_y, step_x, step_y
        """
        x = shape[0]
        y = shape[1]
        n = m = 1
        while x > n * p_size:
            n += 1
        while p_size - 1.0 * (x - p_size) / (n - 1) < 100:
            n += 1
        while y > m * p_size:
            m += 1
        while p_size - 1.0 * (y - p_size) / (m - 1) < 100:
            m += 1
        return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

    def crop_image(self, image, top, left, p_size):
        zero_patch = image[0][:, top:top + p_size, left:left + p_size]
        return zero_patch

    def bbox_all_in_patch(self, bbox, left, top, right, bottom):
        bbox = ((bbox.cpu()).numpy()).tolist()
        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]
        if box_left > left and box_right < right and box_top > top and box_bottom < bottom:
            return True
        else:
            return False

    def bbox_to_patch(self, bbox, left, top, right, bottom, x, y, step_x, step_y):
        bbox = ((bbox.cpu()).numpy()).tolist()
        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]

        left = int(np.round(y * step_y))
        top = int(np.round(x * step_x))

        x1 = (box_left if box_left > left else left) - y * step_y
        y1 = (box_top if box_top > top else top) - x * step_x
        x2 = (box_right if box_right < right else right) - y * step_y
        y2 = (box_bottom if box_bottom < bottom else bottom) - x * step_x
        return [x1, y1, x2, y2]

    def global_to_patch(self, images, p_size):
        """
        image/label => patches
        p_size: patch size
        return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
        """
        len_image = len(images)

        patches = []
        coordinates = []
        templates = []
        sizes = []
        ratios = [(0, 0)] * len_image
        patch_ones = np.ones(p_size)

        images = [images]
        for i in range(len_image):
            h, w = (images[i].shape[2], images[i].shape[3])
            # w, h = images[i].size
            size = (h, w)
            sizes.append(size)
            ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
            template = np.zeros(size)
            n_x, n_y, step_x, step_y = self.get_patch_info(size, p_size[0])  # (11, 11, 449.2, 449.2)
            patches.append([images[i]] * (n_x * n_y))
            coordinates.append([(0, 0)] * (n_x * n_y))
            for x in range(n_x):
                if x < n_x - 1:
                    top = int(np.round(x * step_x))
                else:
                    top = size[0] - p_size[0]
                for y in range(n_y):
                    if y < n_y - 1:
                        left = int(np.round(y * step_y))
                    else:
                        left = size[1] - p_size[1]
                    template[top:top + p_size[0],
                    left:left + p_size[1]] += patch_ones
                    coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])

                    zero_patch = self.crop_image(images[i], top, left, p_size[0])
                    patches[i][
                        x * n_y + y] = zero_patch

            templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda())
        return patches, coordinates, templates, sizes, ratios

    def label_to_patch(self, img, p_size, gt_bboxes, gt_labels, gt_masks):
        len_image = len(img)
        size = (img.shape[2], img.shape[3])
        patch_bboxes = []
        patch_labels = []
        one_patch = []
        one_label = []

        for i in range(len_image):
            n_x, n_y, step_x, step_y = self.get_patch_info(size, p_size[0])
            for x in range(n_x):
                for y in range(n_y):
                    top = int(np.round(x * step_x))
                    left = int(np.round(y * step_y))
                    bottom = int(top + p_size[0])
                    right = int(left + p_size[1])

                    for bbox, label in zip(gt_bboxes[0], gt_labels[0]):
                        if self.bbox_all_in_patch(bbox, left, top, right, bottom):  # bbox在patch范围内
                            one_patch.append(self.bbox_to_patch(bbox, left, top, right, bottom, x, y, step_y, step_y))
                            one_label.append(label)

                    patch_bboxes.append(torch.tensor(one_patch).cuda())
                    patch_labels.append(torch.tensor(one_label).cuda())
        return patch_bboxes, patch_labels

    def update_loss(self, total_losses, loss_part):
        for k, _ in loss_part.items():
            if k not in [k for k, _ in total_losses.items()]:  # 不存在这个loss就加上
                total_losses.update(loss_part)
                return total_losses
            else:  # 存在就相加
                if isinstance(loss_part[k], list):
                    total_losses[k] = [x + y for x, y in list(zip(total_losses[k], loss_part[k]))]
                else:
                    total_losses[k] = total_losses[k] + loss_part[k]
        return total_losses

    def loss_mean(self, total_loss, batch_size):
        for k, v in total_loss.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    total_loss[k][i] = total_loss[k][i] / float(batch_size)
            else:
                total_loss[k] = total_loss[k] / float(batch_size)
        return total_loss

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

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels, gt_masks)  # 将label切分
        del img
        del gt_bboxes
        del gt_labels
        del templates

        idx = []
        count_patch = 0
        losses = dict()
        batch_size = 4

        img_metas[0]['img_shape'] = (800, 800, 3)
        img_metas[0]['pad_shape'] = (800, 800, 3)
        img_metas[0]['scale_factor'] = 1.0

        while count_patch < batch_size:
            i_patch = random.randint(0, len(coordinates[0]) - 1)
            if label_patches[i_patch].shape == torch.Size([0]) or bbox_patches[i_patch] is None or label_patches[
                i_patch] is None:
                continue
            else:
                idx.append(i_patch)
                count_patch += 1

        patches = [patches[0][i] for i in idx]
        bbox_patches = [bbox_patches[i] for i in idx]
        label_patches = [label_patches[i] for i in idx]

        for i, (input_patch, input_bbox, input_label) in enumerate(zip(patches, bbox_patches, label_patches)):
            input_patch = input_patch.unsqueeze(0)
            input_bbox = input_bbox.unsqueeze(0)
            input_label = input_label.unsqueeze(0)

            feat_neck = self.extract_feat(input_patch)
            ########################################################
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    feat_neck,
                    img_metas,
                    input_bbox,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses = self.update_loss(losses, rpn_losses)
            else:
                proposal_list = proposals
            ########################################################
            roi_losses = self.roi_head.forward_train(feat_neck, img_metas, proposal_list,
                                                     input_bbox, input_label,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses = self.update_loss(losses, roi_losses)
            ########################################################
        losses = self.loss_mean(losses, batch_size)

        return losses

    def forward_train_other(self,
                            img,
                            img_metas,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore=None,
                            gt_masks=None,
                            proposals=None,
                            **kwargs):
        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels, gt_masks)  # 将label切分

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
            ########################################################
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    feat_neck,
                    img_metas,
                    input_bbox,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses = self.update_loss(losses, rpn_losses)
            else:
                proposal_list = proposals
            ########################################################
            roi_losses = self.roi_head.forward_train(feat_neck, img_metas, proposal_list,
                                                     input_bbox, input_label,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses = self.update_loss(losses, roi_losses)
            ########################################################
            count_patch += 1
        losses = self.loss_mean(losses, batch_size)

        return losses

    def patch_to_global(self, bbox_result, i_patch):
        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), self.p_size[0])

        for i in range(bbox_result[0].shape[0]):
            bbox_result[0][i][0] += int(i_patch % n_y) * step_y
            bbox_result[0][i][2] += int(i_patch % n_y) * step_y
            bbox_result[0][i][1] += int(i_patch / n_y) * step_x
            bbox_result[0][i][3] += int(i_patch / n_y) * step_x
        return bbox_result

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        img_metas[0]['img_shape'] = (800, 800, 3)
        img_metas[0]['scale_factor'] = 1.0
        img_metas[0]['pad_shape'] = (800, 800, 3)

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
                proposal_list = self.simple_test_rpn(
                    feat_neck, img_metas, self.test_cfg.rpn)
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
                feat_neck, proposal_list, img_metas, rescale=rescale
            )

            bbox_results = self.patch_to_global(bbox_results, i_patch)

            if i_patch > 0:
                result[0] = np.concatenate([result[0], bbox_results[0]])
            else:
                result.extend(bbox_results)
            i_patch += 1

        return result
