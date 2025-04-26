# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='5'

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')

from pysot.core.config import cfg
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox, get_rgbt_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='VTUAV',
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k RGB_T234 GTOT LasHeR数据集名称
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='Dul_SiamCAR_4/checkpoint_e19.pth',
        help='snapshot of models to eval')
parser.add_argument('--model_name', type=str, default='Dul_6/ST_001/test_18',
        help='model_name')

parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml',
        help='config file')

args = parser.parse_args()

torch.set_num_threads(16) # 设置线程数


def main():
    # load config
    cfg.merge_from_file(args.config) # 使用args.config文件中的超参数

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset) # 获取cfg.HP_SEARCH中args.dataset的参数，不同数据集，参数不同
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]} # 赋参数

    # GTOT
    if args.dataset == 'GTOT':
        cur_dir = '/root/cjm/object_tracker/My-/BAN-SiamCAR'
        dataset_root = os.path.join(cur_dir,'test_dataset', args.dataset)

    # RGBT234
    elif args.dataset == 'RGB_T234':
        cur_dir = '/root/cjm/object_tracker/My-/Dul-SiamCAR'
        dataset_root = os.path.join(cur_dir,'test_datasets', args.dataset) #找到图片文件和.json文件

    #LasHer
    elif args.dataset == 'LasHeR':
        cur_dir = '/root/cjm/object_tracker/My-'
        dataset_root = os.path.join(cur_dir,'LasHeR_testdataset', args.dataset) #找到图片文件和.json文件

    elif args.dataset == 'VTUAV':
        cur_dir = '/root/cjm/object_tracker/My-'
        dataset_root = os.path.join(cur_dir,'VTUAV_testdataset', 'test_ST_013')

    else:
        raise Exception("unkonw dataset {}".format(args.dataset))
    #data
    # cur_dir = '/root/cjm/object_tracker/My-'
    # dataset_root = os.path.join(cur_dir,'LasHeR_testdataset', args.dataset) #找到图片文件和.json文件

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False) # 返回初始帧锚框信息，图片名，所有标签，加载的图片

    model_name = args.model_name
    tttt = 0
    tttttt = 0
# model_name：pretrained0.4_0.2_0.3
    # OPE tracking
    for v_idx, video in enumerate(dataset): # 重排序 返回self.videos[key]
        if args.video != '': # 测试一个特别的视频文件，一般不执行
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        track_times = []
        for idx, (img_v, gt_bbox_v, img_i, gt_bbox_i, gt_bbox_init) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                # GTOT dataset
                if args.dataset == 'GTOT':
                    cx_v, cy_v, w_v, h_v = get_rgbt_aligned_bbox(np.array(gt_bbox_v)) # [x,y,w,h]->[Cx,Cy,W,H]
                    gt_bbox_v = [cx_v-(w_v-1)/2, cy_v-(h_v-1)/2, w_v, h_v] # 坐标转换[Cx,Cy,W,H]->[L,T,W,H] 为什么要转来转去的，不直接gt_bbox_=gt_bbox

                    cx_i, cy_i, w_i, h_i = get_rgbt_aligned_bbox(np.array(gt_bbox_i))
                    gt_bbox_i = [cx_i-(w_i-1)/2, cy_i-(h_i-1)/2, w_i, h_i]
                    #
                    cx_init, cy_init, w_init, h_init = get_axis_aligned_bbox(np.array(gt_bbox_init))
                    gt_bbox_init = [cx_init-(w_init-1)/2, cy_init-(h_init-1)/2, w_init, h_init]

                # RGBT234 dataset
                elif args.dataset == 'RGB_T234':
                    cx_v, cy_v, w_v, h_v = get_axis_aligned_bbox(np.array(gt_bbox_v)) # [x,y,w,h]->[Cx,Cy,W,H]
                    gt_bbox_v = [cx_v-(w_v-1)/2, cy_v-(h_v-1)/2, w_v, h_v] # 坐标转换[Cx,Cy,W,H]->[L,T,W,H] 为什么要转来转去的，不直接gt_bbox_=gt_bbox

                    cx_i, cy_i, w_i, h_i = get_axis_aligned_bbox(np.array(gt_bbox_i))
                    gt_bbox_i = [cx_i-(w_i-1)/2, cy_i-(h_i-1)/2, w_i, h_i]

                    cx_init, cy_init, w_init, h_init = get_axis_aligned_bbox(np.array(gt_bbox_init))
                    gt_bbox_init = [cx_init-(w_init-1)/2, cy_init-(h_init-1)/2, w_init, h_init]

                # LasHeR dataset
                elif args.dataset == 'LasHeR':
                    cx_v, cy_v, w_v, h_v = get_axis_aligned_bbox(np.array(gt_bbox_v)) # [x,y,w,h]->[Cx,Cy,W,H]
                    gt_bbox_v = [cx_v-(w_v-1)/2, cy_v-(h_v-1)/2, w_v, h_v] # 坐标转换[Cx,Cy,W,H]->[L,T,W,H] 为什么要转来转去的，不直接gt_bbox_=gt_bbox

                    cx_i, cy_i, w_i, h_i = get_axis_aligned_bbox(np.array(gt_bbox_i))
                    gt_bbox_i = [cx_i-(w_i-1)/2, cy_i-(h_i-1)/2, w_i, h_i]
                    #
                    cx_init, cy_init, w_init, h_init = get_axis_aligned_bbox(np.array(gt_bbox_init))
                    gt_bbox_init = [cx_init-(w_init-1)/2, cy_init-(h_init-1)/2, w_init, h_init]

                elif args.dataset == 'VTUAV':
                    cx_v, cy_v, w_v, h_v = get_axis_aligned_bbox(np.array(gt_bbox_v)) # [x,y,w,h]->[Cx,Cy,W,H]
                    gt_bbox_v = [cx_v-(w_v-1)/2, cy_v-(h_v-1)/2, w_v, h_v] # 坐标转换[Cx,Cy,W,H]->[L,T,W,H] 为什么要转来转去的，不直接gt_bbox_=gt_bbox

                    cx_i, cy_i, w_i, h_i = get_axis_aligned_bbox(np.array(gt_bbox_i))
                    gt_bbox_i = [cx_i-(w_i-1)/2, cy_i-(h_i-1)/2, w_i, h_i]
                    #
                    cx_init, cy_init, w_init, h_init = get_axis_aligned_bbox(np.array(gt_bbox_init))
                    gt_bbox_init = [cx_init-(w_init-1)/2, cy_init-(h_init-1)/2, w_init, h_init]

                else:
                    raise Exception("unkonw dataset {}".format(args.dataset))
                # data dataset
                # cx_v, cy_v, w_v, h_v = get_rgbt_aligned_bbox(np.array(gt_bbox_v)) # [x,y,w,h]->[Cx,Cy,W,H]
                # gt_bbox_v = [cx_v-(w_v-1)/2, cy_v-(h_v-1)/2, w_v, h_v] # 坐标转换[Cx,Cy,W,H]->[L,T,W,H] 为什么要转来转去的，不直接gt_bbox_=gt_bbox
                #
                # cx_i, cy_i, w_i, h_i = get_rgbt_aligned_bbox(np.array(gt_bbox_i))
                # gt_bbox_i = [cx_i-(w_i-1)/2, cy_i-(h_i-1)/2, w_i, h_i]
                # #
                # cx_init, cy_init, w_init, h_init = get_axis_aligned_bbox(np.array(gt_bbox_init))
                # gt_bbox_init = [cx_init-(w_init-1)/2, cy_init-(h_init-1)/2, w_init, h_init]

                tracker.init(img_v, img_i, gt_bbox_init, gt_bbox_v, gt_bbox_i)
                pred_bbox = gt_bbox_init
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img_v, img_i, hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0: # 追踪可视化
                if not any(map(math.isnan,gt_bbox_init)):
                    gt_bbox_init = list(map(int, gt_bbox_init))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img_v, (gt_bbox_init[0], gt_bbox_init[1]),
                                  (gt_bbox_init[0]+gt_bbox_init[2], gt_bbox_init[1]+gt_bbox_init[3]), (0, 255, 0), 3)
                    cv2.rectangle(img_v, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img_v, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img_v)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))
        ttt = idx / toc
        tttt = tttt + ttt
        tttttt = tttttt+1
        print(tttttt)
        print(tttt/tttttt)
    os.chdir(model_path)
    save_file = '../%s' % dataset
    shutil.make_archive(save_file, 'zip')
    print('Records saved at', save_file + '.zip')
    print(tttt / tttttt)


if __name__ == '__main__':
    main()
