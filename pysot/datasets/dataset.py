# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, init_anno,frame_range, num_use, start_idx):
        cur_path = r'/root/cjm/object_tracker/My-/BAN-SiamCAR'
        self.name = name
        self.root = root
        # self.rgb_anno = os.path.join(cur_path, 'tools/', rgb_anno)
        # self.t_anno = os.path.join(cur_path, 'tools/', t_anno)
        self.anno = os.path.join(cur_path, 'tools/', init_anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        # with open(self.rgb_anno, 'r') as f:
        #     meta_data_rgb = json.load(f)
        #     meta_data_rgb = self._filter_zero(meta_data_rgb)
        #
        # with open(self.t_anno, 'r') as f:
        #     meta_data_t = json.load(f)
        #     meta_data_t = self._filter_zero(meta_data_t)

        with open(self.anno, 'r') as f:
            meta_data_init = json.load(f)
            meta_data_init = self._filter_zero(meta_data_init)

        # for video in list(meta_data_rgb.keys()):
        #     for track in meta_data_rgb[video]:
        #         frames = meta_data_rgb[video][track]
        #         frames = list(map(int,
        #                       filter(lambda x: x.isdigit(), frames.keys())))
        #         frames.sort()
        #         meta_data_rgb[video][track]['frames'] = frames
        #         if len(frames) <= 0:
        #             logger.warning("{}/{} has no frames".format(video, track))
        #             del meta_data_rgb[video][track]
        #
        # for video in list(meta_data_rgb.keys()):
        #     if len(meta_data_rgb[video]) <= 0:
        #         logger.warning("{} has no tracks".format(video))
        #         del meta_data_rgb[video]
        #
        #
        # for video in list(meta_data_t.keys()):
        #     for track in meta_data_t[video]:
        #         frames = meta_data_t[video][track]
        #         frames = list(map(int,
        #                       filter(lambda x: x.isdigit(), frames.keys())))
        #         frames.sort()
        #         meta_data_t[video][track]['frames'] = frames
        #         if len(frames) <= 0:
        #             logger.warning("{}/{} has no frames".format(video, track))
        #             del meta_data_t[video][track]
        #
        # for video in list(meta_data_t.keys()):
        #     if len(meta_data_t[video]) <= 0:
        #         logger.warning("{} has no tracks".format(video))
        #         del meta_data_t[video]


        for video in list(meta_data_init.keys()):
            for track in meta_data_init[video]:
                frames = meta_data_init[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data_init[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data_init[video][track]

        for video in list(meta_data_init.keys()):
            if len(meta_data_init[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data_init[video]

        # self.rgb_labels = meta_data_rgb
        # self.t_labels = meta_data_t
        self.labels = meta_data_init
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data_init.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format_rgb = '{}.{}.{}.jpg'
        self.path_format_t = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format_rgb, self.path_format_t))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path_rgb = os.path.join(self.root, video, 'v',
                                  self.path_format_rgb.format(frame, track, 'x'))
        image_path_t = os.path.join(self.root, video, 'i',
                                  self.path_format_t.format(frame, track, 'x'))

        image_anno = self.labels[video][track][frame]
        # image_anno_t = self.t_labels[video][track][frame]
        return image_path_rgb, image_anno, image_path_t

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        # video = self.rgb_labels[video_name]
        video = self.labels[video_name]
        #video_t = self.t_labels[video_name]
        track = np.random.choice(list(video.keys()))
        # track_info = video[track]
        track_info = video[track]
        # track_info_t = video_t[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.init_ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )

        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        # template_image = cv2.imread(template[0])
        # search_image = cv2.imread(search[0])
        template_image_rgb = cv2.imread(template[0])
        template_image_t = cv2.imread(template[2])
        search_image_rgb = cv2.imread(search[0])
        search_image_t = cv2.imread(search[2])
        if template_image_rgb is None:
            print('error image:',template[0])

        if template_image_t is None:
            print('error image:',template[2])

        # get bounding box
        template_box = self._get_bbox(template_image_rgb, template[1])
        # template_box_t = self._get_bbox(template_image_t, template[3])
        search_box = self._get_bbox(search_image_rgb, search[1])
        # search_box_t = self._get_bbox(search_image_t, search[3])


        # augmentation
        template_rgb, template_t, _ = self.template_aug(template_image_rgb, template_image_t,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)


        search_rgb, search_t, bbox = self.search_aug(search_image_rgb, search_image_t,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)


        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template_rgb = template_rgb.transpose((2, 0, 1)).astype(np.float32)
        template_t = template_t.transpose((2, 0, 1)).astype(np.float32)
        search_rgb = search_rgb.transpose((2, 0, 1)).astype(np.float32)
        search_t = search_t.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template_rgb': template_rgb,
                'template_t': template_t,
                'search_rgb': search_rgb,
                'search_t': search_t,
                'label_cls': cls,
                'bbox': np.array([bbox.x1,bbox.y1,bbox.x2,bbox.y2])
                }

