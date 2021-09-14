import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from torch.autograd import Variable
import numpy as np
import random
import mmcv
from mmdet.core.visualization import imshow_det_bboxes


@DETECTORS.register_module()
class TwoStageDetectorLocal(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

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
        super(TwoStageDetectorLocal, self).__init__()
        self.backbone = build_backbone(backbone)

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

        self.p_size = p_size
        self.batch_size = batch_size
        self.ori_shape = ori_shape

    def forward_train(self,
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

        img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['scale_factor'] = 1.0

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
            input_bbox = [input_bbox]

            input_label = label_patches[i_patch]
            input_label = [input_label]

            feat_neck = self.extract_feat(input_patch)
            ##############################################################################
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
            ##############################################################################
            roi_losses = self.roi_head.forward_train(feat_neck, img_metas, proposal_list,
                                                     input_bbox, input_label,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses = self.update_loss(losses, roi_losses)
            ##############################################################################
            count_patch += 1
        losses = self.loss_mean(losses, batch_size)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
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
                proposal_list = self.rpn_head.simple_test_rpn(feat_neck, img_metas)
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

            bbox_results = self.patch_to_global(bbox_results[0], i_patch)

            if i_patch > 0:
                result[0] = np.concatenate([result[0], bbox_results[0]])
            else:
                result.extend(bbox_results)
            i_patch += 1

        return result

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        ########################################################
        width, height = img.shape[1], img.shape[0]

        bbox_result[:, 0] = bbox_result[:, 0] * (width / self.ori_shape[0])
        bbox_result[:, 1] = bbox_result[:, 1] * (height / self.ori_shape[1])
        bbox_result[:, 2] = bbox_result[:, 2] * (width / self.ori_shape[0])
        bbox_result[:, 3] = bbox_result[:, 3] * (height / self.ori_shape[1])

        bbox_result = [bbox_result]
        ########################################################
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)
        # backbone
        input_patch = patches[0][0]
        input_patch = input_patch.unsqueeze(0)
        x = self.extract_feat(input_patch)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

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
        super(TwoStageDetectorLocal, self).init_weights(pretrained)
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
        x = self.backbone(img)  # 4
        # for elem in x:
        #    print("extract_feat shape after backbone", elem.shape)
        c2, c3, c4, c5 = x
        if self.with_neck:
            x = self.neck([c2, c3, c4, c5])
            # x = self.neck(x)
        return x

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
        # zero_patch = torch.zeros((3, p_size, p_size))

        zero_patch = image[0][:, top:top + p_size, left:left + p_size]
        # print(zero_patch.shape)
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
        # print(images.shape)
        # images = transforms.ToPILImage()(images[0].cpu()).convert('RGB')   #不能转Image
        images = [images]
        for i in range(len_image):
            h, w = (images[i].shape[2], images[i].shape[3])
            # w, h = images[i].size
            size = (h, w)
            sizes.append(size)
            ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
            template = np.zeros(size)
            n_x, n_y, step_x, step_y = self.get_patch_info(size, p_size[0])  # (11, 11, 449.2, 449.2)
            # print(n_x, n_y, step_x, step_y)
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
                    # print(top,left)

                    zero_patch = self.crop_image(images[i], top, left, p_size[0])
                    patches[i][
                        x * n_y + y] = zero_patch  # transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
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

        left = int(np.round(y * step_y))
        top = int(np.round(x * step_x))

        x1 = (box_left if box_left > left else left) - y * step_y
        y1 = (box_top if box_top > top else top) - x * step_x
        x2 = (box_right if box_right < right else right) - y * step_y
        y2 = (box_bottom if box_bottom < bottom else bottom) - x * step_x
        # return torch.tensor([x1,y1,x2,y2]).cuda()
        return [x1, y1, x2, y2]

    def label_to_patch(self, img, p_size, gt_bboxes, gt_labels, gt_masks):
        # print((gt_labels[0]))
        len_image = len(img)
        size = (img.shape[2], img.shape[3])
        patch_bboxes = []
        patch_labels = []  # label都是1
        patch_masks = []  # mask像图片一样切一下就行
        one_patch = []
        one_label = []
        # patches, coordinates, templates, sizes, ratios = \
        #    self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        for i in range(len_image):  # for batch size
            n_x, n_y, step_x, step_y = self.get_patch_info(size, p_size[0])
            for x in range(n_x):
                for y in range(n_y):
                    top = int(np.round(x * step_x))
                    left = int(np.round(y * step_y))
                    bottom = int(top + p_size[0])
                    right = int(left + p_size[1])

                    for bbox, label in zip(gt_bboxes[0], gt_labels[0]):
                        # print(bbox)
                        if self.bbox_all_in_patch(bbox, left, top, right, bottom):  # bbox在patch范围内
                            one_patch.append(self.bbox_to_patch(bbox, left, top, right, bottom, x, y, step_y, step_y))
                            one_label.append(label)

                    patch_bboxes.append(torch.tensor(one_patch).cuda())
                    patch_labels.append(torch.tensor(one_label).cuda())
                    one_label = []
                    one_patch = []
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

    def patch_to_global(self, bbox_result, i_patch):
        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), self.p_size[0])

        for i in range(bbox_result[0].shape[0]):
            bbox_result[0][i][0] += int(i_patch % n_y) * step_y
            bbox_result[0][i][2] += int(i_patch % n_y) * step_y
            bbox_result[0][i][1] += int(i_patch / n_y) * step_x
            bbox_result[0][i][3] += int(i_patch / n_y) * step_x
        return bbox_result

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
