from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import numpy as np
from torch.autograd import Variable
import torch
import random


@DETECTORS.register_module()
class SparseRCNN_Local(TwoStageDetector):
    r"""Implementation of `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self, *args, **kwargs):
        super(SparseRCNN_Local, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'Sparse R-CNN do not support external proposals'
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

    def label_to_patch(self, img, p_size, gt_bboxes, gt_labels):
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
            # print("k:",k,type(loss_part[k]))
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
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert proposals is None, 'Sparse R-CNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'Sparse R-CNN does not instance segmentation'

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels)

        img_metas[0]['img_shape'] = (800, 800, 3)
        img_metas[0]['pad_shape'] = (800, 800, 3)
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
            input_label = label_patches[i_patch]

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

        # x = self.extract_feat(img)
        # proposal_boxes, proposal_features, imgs_whwh = \
        #     self.rpn_head.forward_train(x, img_metas)
        # roi_losses = self.roi_head.forward_train(
        #     x,
        #     proposal_boxes,
        #     proposal_features,
        #     img_metas,
        #     gt_bboxes,
        #     gt_labels,
        #     gt_bboxes_ignore=gt_bboxes_ignore,
        #     gt_masks=gt_masks,
        #     imgs_whwh=imgs_whwh)
        # return roi_losses

    def patch_to_global(self, bbox_result, i_patch):
        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), self.p_size[0])

        for i in range(bbox_result[0].shape[0]):
            bbox_result[0][i][0] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][2] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][1] += int((i_patch) / n_y) * step_x
            bbox_result[0][i][3] += int((i_patch) / n_y) * step_x
        return bbox_result

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        assert self.with_bbox, "Bbox head must be implemented."
        img_metas[0]['img_shape'] = (800, 800, 3)
        img_metas[0]['scale_factor'] = 1.0
        img_metas[0]['pad_shape'] = (800, 800, 3)

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)

        i_patch = 0
        result = []
        return_rpn = False

        for i in range(len(coordinates[0])):
            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)
            feat_neck = self.extract_feat(input_patch)

        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bbox_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas)
        return roi_outs
