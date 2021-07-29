from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import mmcv
import argparse
from mmdet.datasets import build_dataloader, build_dataset
import os
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from tools.test import single_gpu_test
import time
import copy
import numpy as np
import datetime
import shutil


def computeIoU(self, imgId, catId):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]

    if p.iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif p.iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')

    # compute iou between each dt and gt region
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d, g, iscrowd)
    return ious


def evaluateImg(self, imgId, catId, aRng, score, prog_bar):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return None

    for g in gt:
        if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
            g['_ignore'] = 1
        else:
            g['_ignore'] = 0

    # sort dt highest score first, sort gt ignore last
    gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
    gt = [gt[i] for i in gtind]

    #################################################################################
    dt = [d for d in dt if d['score'] > score]
    #################################################################################

    dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in dtind[0:p.maxDets[-1]]]
    iscrowd = [int(o['iscrowd']) for o in gt]
    # load computed ious
    ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

    T = len(p.iouThrs)
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))
    gtIg = np.array([g['_ignore'] for g in gt])
    dtIg = np.zeros((T, D))
    if not len(ious) == 0:
        for tind, t in enumerate(p.iouThrs):
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    # if this gt already matched, and not a crowd, continue
                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']
    # set unmatched detections outside of area range to ignore
    a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
    dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
    # store results for given image and category
    prog_bar.update()
    return {
        'image_id': imgId,
        'category_id': catId,
        'aRng': aRng,
        'score': score,
        'maxDet': p.maxDets[-1],  # 100000
        'dtIds': [d['id'] for d in dt],
        'gtIds': [g['id'] for g in gt],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dt],
        'gtIgnore': gtIg,
        'dtIgnore': dtIg,
    }


def coco_evaluate(self):
    tic = time.time()
    print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))

    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    self.ious = {(imgId, catId): computeIoU(self, imgId, catId) \
                 for imgId in p.imgIds
                 for catId in catIds}

    prog_bar = mmcv.ProgressBar(len(self.params.imgIds))
    self.evalImgs = [evaluateImg(self, imgId, catId, p.areaRng[0], p.scoceThrs, prog_bar)
                     for catId in catIds
                     for imgId in p.imgIds
                     ]
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def coco_accumulate(self, p=None):
    print('Accumulating evaluation results...')
    tic = time.time()
    if not self.evalImgs:
        print('Please run evaluate() first')
    # allows input customized parameters
    if p is None:
        p = self.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]
    T = len(p.iouThrs)
    R = len(p.recThrs)
    K = len(p.catIds) if p.useCats else 1
    A = len(p.areaRng)
    M = len(p.maxDets)
    precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))
    Pre = 0

    # create dictionary for future indexing
    _pe = self._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)
    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds) if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0 * A0 * I0
        for a, a0 in enumerate(a_list):
            Na = a0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    Pre = np.mean(pr)
                    q = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, k, a, m] = rc[-1]
                    else:
                        recall[t, k, a, m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t, :, k, a, m] = np.array(q)
                    scores[t, :, k, a, m] = np.array(ss)
    self.eval = {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'Pr': Pre,
        'recall': recall,
        'scores': scores,
    }
    toc = time.time()
    print('DONE (t={:0.2f}s).'.format(toc - tic))


def coco_summarize(self, args):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100000, write_handle=None):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | Score={:>4} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        if ap == 1:
            titleStr = 'Average Precision'
        elif ap == 0:
            titleStr = 'Average Recall'
        elif ap == 2:
            titleStr = 'Precision'
        if ap == 1:
            typeStr = '(AP)'
        elif ap == 0:
            typeStr = '(AR)'
        elif ap == 2:
            typeStr = '(PR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif ap == 0:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        elif ap == 2:
            mean_s = self.eval['Pr']

        if ap != 2:
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, p.scoceThrs, areaRng, maxDets, mean_s))
        if write_handle is not None:
            write_handle.write(iStr.format(titleStr, typeStr, iouStr, p.scoceThrs, areaRng, maxDets, mean_s) + '\n')
        return mean_s

    def _summarizeDets(args):
        stats = np.zeros((3,))
        f = open(os.path.join(args.work_dir, 'result.txt'), 'a+')
        bare_name = os.path.basename(args.work_dir)
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        stats[0] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[1] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        stats[2] = _summarize(2, iouThr=.5, maxDets=self.params.maxDets[-1], write_handle=f)
        f.write('\n')
        f.close()
        print('Successfully Write Result File...')

        # for batter stored result file,create a dir to stored result,and it can be upload to github
        assert os.path.isfile(os.path.join(args.work_dir, 'result.txt'))
        out_dir = 'result'
        os.makedirs(out_dir, exist_ok=True)
        out_file_name = os.path.join(out_dir, '{}.txt'.format(bare_name))
        if os.path.isfile(out_file_name):
            shutil.rmtree(out_file_name)
        shutil.copy(os.path.join(args.work_dir, 'result.txt'), out_file_name)
        return stats

    summarize = _summarizeDets
    self.stats = summarize(args)


def evaluate(args, anns):
    cocoGt = COCO(args.val_path)
    cocoDt = cocoGt.loadRes(anns)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=args.eval)

    set_param(cocoEval)
    coco_evaluate(cocoEval)
    coco_accumulate(cocoEval)
    coco_summarize(cocoEval, args)


def det2json(dataset, results, mode):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        if mode == 'local':
            imgInfo = dataset.data_infos[idx]
            width = imgInfo['width']
            height = imgInfo['height']
            result[:, 0] = result[:, 0] * (width / 3000)
            result[:, 1] = result[:, 1] * (height / 3000)
            result[:, 2] = result[:, 2] * (width / 3000)
            result[:, 3] = result[:, 3] * (height / 3000)
            for i in result:
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = dataset.xyxy2xywh(i[:-1])
                data['score'] = float(i[4])
                data['category_id'] = 1
                json_results.append(data)
        elif mode == 'global':
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = dataset.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = dataset.cat_ids[label]
                    json_results.append(data)
    return json_results


def det(args):
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=False,
                                   shuffle=False)

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, data_loader, show=args.show, out_dir=args.outdir)
    anns = det2json(dataset, outputs, args.mode)
    mmcv.dump(anns, args.ResFile)

    return anns


def set_param(self):
    p = self.params

    ########################################################################################
    # 调用cocoeval时所需的参数
    p.maxDets = [100000]
    p.iouThrs = np.array([0.5])
    p.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    p.scoceThrs = args.score
    p.areaRng = [[0 ** 2, 1e5 ** 2]]
    ########################################################################################

    self.params = p


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument('--work_dir', default='work_dirs/faster_local')
    parser.add_argument('--score', default=0.3, type=float)
    parser.add_argument('--show', default=False, type=bool)

    parser.add_argument('--val_path', type=str, default='data/xview/annotations/instances_val2017.json')
    parser.add_argument('--eval', type=str, default='bbox', nargs='+',
                        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
                        help='eval types')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    bare_name = os.path.basename(args.work_dir)
    if 'global' in bare_name:
        args.mode = 'global'
    elif 'local' in bare_name:
        args.mode = 'local'
    else:
        raise ValueError('Wrong work_dir name')
    print('Start Evaluateing {} model'.format(bare_name))

    args.config = os.path.join(args.work_dir, '{}.py'.format(bare_name))
    args.checkpoint = os.path.join(args.work_dir, 'epoch_50.pth')
    args.ResFile = os.path.join(args.work_dir, 'ResFile.json')
    if args.show is True:
        args.outdir = os.path.join(args.work_dir, 'out')
        os.makedirs(args.outdir, exist_ok=True)
    else:
        args.outdir = None

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.show or not os.path.isfile(args.ResFile):
        anns = det(args)
        evaluate(args, anns)
    else:
        print('Load ResFile From Local...')
        evaluate(args, args.ResFile)
