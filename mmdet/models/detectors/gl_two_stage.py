from ..builder import DETECTORS, build_head, build_neck
from .base import BaseDetector
import torch.nn as nn
import torch
import mmcv
import numpy as np
from torch.autograd import Variable
import cv2


@DETECTORS.register_module()
class GLTwoStage(BaseDetector):
    def __init__(self,
                 mode,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(GLTwoStage, self).__init__()
        self.MODE = mode

        self.matched_proposal = []
        if 'assess_proposal_quality' in test_cfg:
            self.assess_proposal_quality = test_cfg.assess_proposal_quality
        else:
            self.assess_proposal_quality = False

        if neck is not None:
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
                self.neck.init_weights(self.MODE)
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
        pass

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        pass

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        pass

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        if self.MODE == 1:
            x = self.neck(img, None, None, None, mode=self.MODE)
        elif self.MODE == 2:
            patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
            input_img = patches[0][0].unsqueeze(0)
            input_patch = patches[0][0].unsqueeze(0)
            i_patch = 0
            batch_size = 2
            x = self.neck(input_img, input_patch,
                          coordinates[0][i_patch: i_patch + 1], ratios[0], templates[0],
                          mode=self.MODE, n_patch=batch_size, i_patch=i_patch, )
        elif self.MODE == 3:
            input_img = torch.zeros(1, 3, self.ori_shape[0], self.ori_shape[1])
            patches, coordinates, templates, sizes, ratios = self.global_to_patch(input_img, self.p_size)
            x = self.neck(img, patches, coordinates, ratios, templates, mode=self.MODE)
        elif self.MODE == 4:
            patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
            input_img = patches[0][0].unsqueeze(0)
            input_patch = patches[0][0].unsqueeze(0)
            i_patch = 0
            batch_size = 2
            x = self.neck(input_img, input_patch,
                          coordinates[0][i_patch: i_patch + 1], ratios[0], templates[0],
                          mode=self.MODE, n_patch=batch_size, i_patch=i_patch, )
        else:
            raise ValueError('wrong mode:{}'.format(self.MODE))
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

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
        while p_size - 1.0 * (x - p_size) / (n - 1) < 50:  # 512
            n += 1
        while y > m * p_size:
            m += 1
        while p_size - 1.0 * (y - p_size) / (m - 1) < 50:  # 512
            m += 1
        return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

    def crop_image(self, image, top, left, p_size):
        # zero_patch = torch.zeros((3, p_size, p_size))
        zero_patch = image[0][:, top:top + p_size, left:left + p_size]
        return zero_patch

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
        # images = transforms.ToPILImage()(images[0].cpu()).convert('RGB')   #不能转Image
        images = [images]
        for i in range(len_image):
            h, w = (images[i].shape[2], images[i].shape[3])
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
                    left:left + p_size[1]] += patch_ones  # 0:508, 449:449+508 patch之间存在（508-449）的重叠
                    coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])  # 449/5000 归一化

                    zero_patch = self.crop_image(images[i], top, left, p_size[0])
                    patches[i][
                        x * n_y + y] = zero_patch
                    # transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
                    # patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])

            templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda())
        return patches, coordinates, templates, sizes, ratios

    def bbox_in_patch(self, bbox, left, top, right, bottom):
        bbox = ((bbox.cpu()).numpy()).tolist()
        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]
        if box_left > right or box_right < left or box_top > bottom or box_bottom < top:
            return False
        else:
            return True

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
        x1 = (box_left if box_left > left else left) - y * step_y  #
        y1 = (box_top if box_top > top else top) - x * step_x  # x*step_x
        x2 = (box_right if box_right < right else right) - y * step_y
        y2 = (box_bottom if box_bottom < bottom else bottom) - x * step_x
        # return torch.tensor([x1,y1,x2,y2]).cuda()
        return [x1, y1, x2, y2]

    def computeOverlap(self, bbox, left, top, right, bottom):
        bbox = ((bbox.cpu()).numpy()).tolist()

        box_left = bbox[0]
        box_top = bbox[1]
        box_right = bbox[2]
        box_bottom = bbox[3]

        area1 = (box_right - box_left + 1) * (box_bottom - box_top + 1)
        area2 = (right - left + 1) * (bottom - top + 1)

        x_start = np.maximum(box_left, left)
        y_start = np.maximum(box_top, top)
        x_end = np.minimum(box_right, right)
        y_end = np.minimum(box_bottom, bottom)

        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        iou = overlap / area1
        return iou

    def label_to_patch(self, img, img_meta, p_size, gt_bboxes, gt_labels, gt_masks):
        len_image = len(img)
        size = (img.shape[2], img.shape[3])

        patch_bboxes = []
        patch_labels = []  # label都是1
        one_patch = []
        one_label = []
        for i in range(len_image):  # for batch size
            n_x, n_y, step_x, step_y = self.get_patch_info(size, p_size[0])
            for x in range(n_x):
                for y in range(n_y):
                    left = int(np.round(y * step_y))
                    top = int(np.round(x * step_x))
                    right = int(left + p_size[1])
                    bottom = int(top + p_size[0])
                    for index, bbox_lable in enumerate(zip(gt_bboxes[0], gt_labels[0])):
                        # if index not in computed:
                        bbox = bbox_lable[0]
                        lable = bbox_lable[1]
                        overlap = self.computeOverlap(bbox, left, top, right, bottom)
                        if overlap > 0.7:
                            one_patch.append(
                                self.bbox_to_patch(bbox, left, top, right, bottom, x, y, step_x, step_y))
                            one_label.append(lable)
                    if len(one_patch) > 150 or len(one_label) > 150:
                        one_patch = one_patch[:150]
                        one_label = one_label[:150]
                    patch_bboxes.append(torch.tensor(one_patch).cuda())
                    patch_labels.append(torch.tensor(one_label).cuda())
                    one_label = []
                    one_patch = []

        return patch_bboxes, patch_labels  # ,overlap

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

    def img_resize(self, img, size):  # resize原图
        # img = (img[0].cpu()).numpy()
        # img = img.transpose(1, 2, 0)
        img = mmcv.imresize(img, size, return_scale=False)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = mmcv.imnormalize(img, mean, std, True)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).cuda()
        img = img.unsqueeze(0)

        return img

    def patch_to_global(self, bbox_result, i_patch):
        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), self.p_size[0])

        for i in range(bbox_result[0].shape[0]):
            bbox_result[0][i][0] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][2] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][1] += int((i_patch) / n_y) * step_x
            bbox_result[0][i][3] += int((i_patch) / n_y) * step_x
        return bbox_result
