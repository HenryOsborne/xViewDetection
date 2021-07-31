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
class LocalGLGA(GLTwoStage):
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
        super(LocalGLGA, self).__init__(
            mode=mode,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
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
            input_img = self.img_resize(input_img, self.p_size)

            img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
            img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
            img_metas[0]['scale_factor'] = 1.0

            count_patch = 0

            valid_patch_index = []
            i_patch_used = []

            for index, box_patch in enumerate(bbox_patches):
                if box_patch.tolist():
                    valid_patch_index.append(index)

            # batch_size = 2 if len(valid_patch_index) > 2 else len(valid_patch_index)
            batch_size = self.batch_size
            while count_patch < batch_size:
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
                    ##########################################################################
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

        ori_shape = img_metas[0]['img_shape']

        img_metas[0]['img_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['pad_shape'] = (self.p_size[0], self.p_size[1], 3)
        img_metas[0]['scale_factor'] = 1.0

        patches, coordinates, templates, sizes, ratios = self.global_to_patch(img, self.p_size)
        # patches,patch位置,？,img_size,p_size/img_size

        original_img = cv2.imread(img_metas[0]['filename'])
        img_value = original_img.astype(np.float32)
        input_img = self.img_resize(img_value, self.p_size)

        show_feature = False
        return_rpn = False

        i_patch = 0
        result = []

        n_x, n_y, step_x, step_y = self.get_patch_info((ori_shape[0],  # 3000
                                                        ori_shape[1]),  # 3000
                                                       self.p_size[0])  # 800
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
            bbox_results = self.patch_to_global(bbox_results[0], i_patch)

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
