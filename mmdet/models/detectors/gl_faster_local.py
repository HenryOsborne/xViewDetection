from ..builder import DETECTORS
from .gl_two_stage import GLTwoStage
import torch
import mmcv
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import random
from mmdet.core.visualization import imshow_det_bboxes


@DETECTORS.register_module
class GLFasterRCNNLocal(GLTwoStage):
    MODE = 2

    def __init__(self,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(GLFasterRCNNLocal, self).__init__(
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

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
        input_img = patches[0][0]
        input_patch = patches[0][0]
        i_patch = 0
        batch_size = 2
        feat_neck = self.neck(input_img, input_patch,
                              coordinates[0][i_patch: i_patch + 1], ratios[0], templates[0],
                              mode=self.MODE, n_patch=batch_size, i_patch=i_patch, )
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(feat_neck)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(feat_neck, proposals)
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
        losses = dict()

        if len(gt_bboxes[0]) > 0 and len(gt_labels[0]) > 0:
            patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
            # patches,patch位置,？,img_size,p_size/img_size
            # overlap
            bbox_patches, label_patches = self.label_to_patch(img,
                                                              img_metas,
                                                              self.p_size,
                                                              gt_bboxes,
                                                              gt_labels,
                                                              gt_masks)  # 将label切分

            input_img = cv2.imread(img_metas[0]['filename'])
            input_img = input_img.astype(np.float32)
            input_img = self.img_resize(input_img, (800, 800))

            img_metas[0]['img_shape'] = (800, 800, 3)
            img_metas[0]['pad_shape'] = (800, 800, 3)
            img_metas[0]['scale_factor'] = 1.0

            count_patch = 0

            valid_patch_index = []
            i_patch_used = []

            for index, box_patch in enumerate(bbox_patches):
                if box_patch.tolist():
                    valid_patch_index.append(index)

            batch_size = 2 if len(valid_patch_index) > 2 else len(valid_patch_index)
            # batch_size=1
            while count_patch < batch_size:  # batch_size:  #len(coordinates[0]):
                if valid_patch_index:
                    try:
                        valid_index = random.randint(0, len(valid_patch_index) - 1)
                        if valid_index not in i_patch_used:
                            i_patch = valid_patch_index[valid_index]
                            i_patch_used.append(valid_index)
                        else:
                            continue
                    except:
                        continue

                    input_patch = patches[0][i_patch]
                    input_patch = input_patch.unsqueeze(0)

                    input_bbox = bbox_patches[i_patch]
                    input_bbox = input_bbox.unsqueeze(0)
                    input_label = label_patches[i_patch]
                    input_label = input_label.unsqueeze(0)

                    feat_neck = self.neck(input_img, input_patch,
                                          coordinates[0][i_patch: i_patch + 1], ratios[0], templates[0],
                                          mode=self.MODE, n_patch=batch_size, i_patch=i_patch, )

                    ##########################################################################
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

                    roi_losses = self.roi_head.forward_train(feat_neck, img_metas, proposal_list,
                                                             input_bbox, input_label,
                                                             gt_bboxes_ignore, gt_masks,
                                                             **kwargs)
                    losses = self.update_loss(losses, roi_losses)
                    ##########################################################################
                    count_patch += 1
            losses = self.loss_mean(losses, batch_size)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        img_metas[0]['img_shape'] = (800, 800, 3)
        img_metas[0]['scale_factor'] = 1.0
        img_metas[0]['pad_shape'] = (800, 800, 3)

        patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
        # patches,patch位置,？,img_size,p_size/img_size

        original_img = cv2.imread(img_metas[0]['filename'])
        img_value = original_img.astype(np.float32)
        input_img = self.img_resize(img_value, (800, 800))

        show_feature = False
        return_rpn = False

        i_patch = 0
        result = []

        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), 1024)  # 800
        step_x /= 4
        step_y /= 4

        for i in range(len(coordinates[0])):
            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)
            feat_neck = self.neck(input_img, input_patch,
                                  coordinates[0][i_patch: i_patch + 1], ratios[0], templates[0],
                                  mode=self.MODE, n_patch=1, i_patch=i_patch, )  # 在这里传入img

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

            bbox_results = self.roi_head.simple_test(feat_neck, proposal_list, img_metas, rescale=rescale)
            bbox_results = self.patch_to_global(bbox_results, i_patch)

            if not self.with_mask:
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

        bbox_result[:, 0] = bbox_result[:, 0] * (width / 3000)
        bbox_result[:, 1] = bbox_result[:, 1] * (height / 3000)
        bbox_result[:, 2] = bbox_result[:, 2] * (width / 3000)
        bbox_result[:, 3] = bbox_result[:, 3] * (height / 3000)

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
