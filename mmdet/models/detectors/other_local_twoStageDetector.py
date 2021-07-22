import torch
import torch.nn as nn
import random
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..builder import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
import numpy as np
from torchvision import transforms
from torch.autograd import Variable


@DETECTORS.register_module
class Other_Local_TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                                   MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Other_Local_TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = builder.build_head(rpn_head_)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        self.p_size = (800, 800)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(Other_Local_TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)  # 4
        # for elem in x:
        #    print("extract_feat shape after backbone", elem.shape)
        c2, c3, c4, c5 = x
        if self.with_neck:
            x = self.neck([c2, c3, c4, c5])
            # x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred,)
        return outs

    def get_patch_info(self, shape, p_size):
        '''
        shape: origin image size, (x, y)
        p_size: patch size (square)
        return: n_x, n_y, step_x, step_y
        '''
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
        '''
        image/label => patches
        p_size: patch size
        return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
        '''
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

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size
        bbox_patches, label_patches = \
            self.label_to_patch(img, self.p_size, gt_bboxes, gt_labels, gt_masks)  # 将label切分

        img_meta[0]['img_shape'] = (800, 800, 3)
        img_meta[0]['pad_shape'] = (800, 800, 3)
        img_meta[0]['scale_factor'] = 1.0

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

            feat_neck = self.extract_feat(input_patch)  # 在这里传入img

            # RPN forward and loss
            if self.with_rpn:
                rpn_outs = self.rpn_head(feat_neck)  # 这里传的x应该有5个：c2，c3，c4，c5，c6

                rpn_loss_inputs = rpn_outs + (input_bbox.unsqueeze(0), img_meta)  # bbox_patches[i_patch].unsqueeze(0)

                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

                losses = self.update_loss(losses, rpn_losses)

                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
                proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            else:
                proposal_list = proposals

            # assign gts and sample proposals
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(proposal_list[i],
                                                         input_bbox,
                                                         gt_bboxes_ignore[i],
                                                         input_label)
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        bbox_patches[i_patch],
                        label_patches[i_patch],
                        feats=[lvl_feat[i][None] for lvl_feat in feat_neck])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            if self.with_bbox:
                rois = bbox2roi([res.bboxes for res in sampling_results])
                # TODO: a more flexible way to decide which feature maps to use
                bbox_feats = self.bbox_roi_extractor(
                    feat_neck[:self.bbox_roi_extractor.num_inputs], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = self.bbox_head(bbox_feats)  # see bbox_head forward

                bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                          input_bbox, input_label,
                                                          self.train_cfg.rcnn)
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois,
                                                *bbox_targets)

                losses = self.update_loss(losses, loss_bbox)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = self.mask_roi_extractor(
                        feat_neck[:self.mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_pred = self.mask_head(mask_feats)

                mask_targets = self.mask_head.get_target(sampling_results,
                                                         gt_masks,
                                                         self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)

            count_patch += 1

        losses = self.loss_mean(losses, batch_size)

        return losses  # 返回相加后平均的loss

    def patch_to_global(self, bbox_result, i_patch):
        n_x, n_y, step_x, step_y = self.get_patch_info((3000, 3000), self.p_size[0])

        for i in range(bbox_result[0].shape[0]):
            bbox_result[0][i][0] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][2] += int((i_patch) % n_y) * step_y
            bbox_result[0][i][1] += int((i_patch) / n_y) * step_x
            bbox_result[0][i][3] += int((i_patch) / n_y) * step_x
        return bbox_result

    """
    import cv2
    import numpy as np
    # image_show = np.uint8((img.cpu()).numpy())
    # image_show = np.squeeze(image_show)
    # image_show = image_show.transpose((1, 2, 0))

    # image_show=cv2.imread('../data/coco/train2017/20.tif')
    # image_show=cv2.resize(image_show,(800,800))
    image_show = patches[0][i_patch]
    image_show = cv2.cvtColor(np.array(image_show), cv2.COLOR_RGB2BGR)
    print(type(image_show))
    for roi in proposal_list[0]:
        roi = (roi.cpu()).numpy()

        image_show = cv2.rectangle(image_show, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 1)
    cv2.imshow("1", image_show)
    cv2.waitKey(0)
    """

    def simple_test(self, img, img_meta, proposals=None, rescale=False):  # local demo用
        """Test without augmentation."""

        assert self.with_bbox, "Bbox head must be implemented."
        img_meta[0]['img_shape'] = (800, 800, 3)
        img_meta[0]['scale_factor'] = 1.0
        img_meta[0]['pad_shape'] = (800, 800, 3)

        patches, coordinates, templates, sizes, ratios = \
            self.global_to_patch(img, self.p_size)  # patches,patch位置,？,img_size,p_size/img_size

        '''                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        f = open('../test_img.txt')
        line = f.readlines()
        line = line[0]
        print(line)
        '''

        show_feature = False
        i_patch = 0
        result = []
        return_rpn = False  # False

        for i in range(len(coordinates[0])):
            # print(i_patch)

            input_patch = patches[0][i_patch]
            input_patch = input_patch.unsqueeze(0)
            feat_neck = self.extract_feat(input_patch)  # 在这里传入img

            ''' 
            if show_feature == True:
                from PIL import Image
                feature = feat_neck[0][0][0].cpu()
                feature = 1.0 / (1 + np.exp(-1 * feature))
                feature = np.round(feature * 255)
                img = transforms.ToPILImage()(feature).convert('RGB')
                img = img.resize((800, 800), Image.ANTIALIAS)
                img.save('./feature/' + str(i_patch) + 'c2.jpg')
            '''

            proposal_list = self.simple_test_rpn(
                feat_neck, img_meta, self.test_cfg.rpn) if proposals is None else proposals

            # print(proposal_list[0].shape)

            if return_rpn:
                gl_proposal = self.patch_to_global(proposal_list, i_patch)
                if i_patch > 0:

                    result[0] = np.concatenate([result[0], gl_proposal[0].cpu().numpy()])
                else:
                    result.extend([gl_proposal[0].cpu().numpy()])
                i_patch += 1
                continue
            #########################  测试rpn是否正常   ########################

            ########################################################
            det_bboxes, det_labels = self.simple_test_bboxes(
                feat_neck, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
            # print(img_meta[0]['ori_shape'])
            ori_shape = img_meta[0]['ori_shape']

            bbox_results = self.patch_to_global(bbox_results, i_patch)

            if not self.with_mask:
                if i_patch > 0:
                    result[0] = np.concatenate([result[0], bbox_results[0]])
                else:
                    result.extend(bbox_results)
                i_patch += 1

                # return bbox_results
            else:
                segm_results = self.simple_test_mask(
                    feat_neck, img_meta, det_bboxes, det_labels, rescale=rescale)
                return bbox_results, segm_results

        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
