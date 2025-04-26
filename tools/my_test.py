# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')
import time

from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox, get_rgbt_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='GTOT',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='Dul_SiamCAR/checkpoint_e10.pth',
        help='snapshot of models to eval')

parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml',
        help='config file')

parser.add_argument('--record_path', type=str, default='Dul_SiamCAR/record_yuan_e10',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(4)


def _record(record_file, boxes, times):
    # record bounding boxes
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')

    # print('  Results recorded at', record_file)

    # record running times
    time_dir = os.path.join(record_dir, 'times')
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)
    time_file = os.path.join(time_dir, os.path.basename(
        record_file).replace('.txt', '_time.txt'))
    np.savetxt(time_file, times, fmt='%.8f')


def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    cur_dir = '/root/cjm/object_tracker/My-/BAN-SiamCAR'
    dataset_root = os.path.join(cur_dir,'test_dataset', args.dataset)

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                             dataset_root=dataset_root,
                                             load_img=False)


    model_name = args.snapshot.split('/')[-2] + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])

    # OPE tracking
    toc = 0

    # track_times = []

    for idx, (rgb_img_files, t_img_files, groundtruth, gt_r, gt_t) in tqdm(enumerate(dataset), total=len(dataset)):
        seq_name = dataset.seq_names[idx]
        record_file = os.path.join(args.record_path, seq_name, '%s.txt' % seq_name)
        frame_num = len(rgb_img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = groundtruth[0]
        times = np.zeros(frame_num)
        pred_bboxes = []
        #tic = cv2.getTickCount()
        #begin = time.time()
        for f, rgb_img_file in enumerate(rgb_img_files):
            t_img_file = t_img_files[f]
            rgb_img_file = rgb_img_files[f]
            rgb_img = cv2.imread(rgb_img_file)
            t_img = cv2.imread(t_img_file)
            begin = time.time()
            if f == 0:
                # groundtruth -> cx, cy, w, h
                cx, cy, w, h = get_axis_aligned_bbox(np.array(groundtruth[0]))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

                # gt_r -> cx, cy, w, h
                cx_r, cy_r, w_r, h_r = get_rgbt_aligned_bbox(np.array(gt_r[0]))
                gt_bbox_r = [cx_r-(w_r-1)/2, cy_r-(h_r-1)/2, w_r, h_r]

                # gt_t -> cx, cy, w, h
                cx_t, cy_t, w_t, h_t = get_rgbt_aligned_bbox(np.array(gt_t[0]))
                gt_bbox_t = [cx_t-(w_t-1)/2, cy_t-(h_t-1)/2, w_t, h_t]

                tracker.init(rgb_img, t_img, gt_bbox_, gt_bbox_r, gt_bbox_t)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                # boxes[f, :] = self.update(rgb_img, t_img)
                outputs = tracker.track(rgb_img, t_img,hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
            times[f] = time.time() - begin
            toc = sum(times)
            if f == 0:
                cv2.destroyAllWindows()
            if args.vis and f > 0:
                if not any(map(math.isnan,groundtruth[f])):
                    gt_bbox = list(map(int, groundtruth[f]))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(rgb_img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(rgb_img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(rgb_img, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(seq_name, rgb_img)
                    cv2.waitKey(1)
        print('(Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            seq_name, toc, len(rgb_img_files) / toc))
        _record(record_file, pred_bboxes, times)


















if __name__ == '__main__':
    main()
