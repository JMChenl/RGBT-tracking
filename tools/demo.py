from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np

sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot/checkpoint_e19.pth', help='model name')
parser.add_argument('--video_name', default='../test_dataset/GTOT/BlackCar', type=str, help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        demo_seq_dir = args.video_name
        sqe_dir_rgb = os.path.join(demo_seq_dir, 'v')
        sqe_dir_t = os.path.join(demo_seq_dir, 'i')
        # anno_files = os.path.join(demo_seq_dir, 'init.txt')

        rgb_image = sorted(glob(os.path.join(sqe_dir_rgb,  '*.pn*')))
        t_image = sorted(glob(os.path.join(sqe_dir_t,  '*.pn*')))

        # anno = np.loadtxt(anno_files, delimiter='\t')

        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for idx ,img in enumerate (rgb_image):
            rgb_images = cv2.imread(rgb_image[idx])
            t_images = cv2.imread(t_image[idx])

            yield rgb_images, t_images


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}



    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for rgb_images, t_images in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, rgb_images, False, False)
            except:
                exit()
            tracker.init(rgb_images, t_images, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(rgb_images, t_images, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(rgb_images, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, rgb_images)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
