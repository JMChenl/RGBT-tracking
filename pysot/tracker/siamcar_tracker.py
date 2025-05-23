# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip
import time


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img_rgb, img_t, bbox, bbox_r, bbox_t):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])

        self.center_pos_r = np.array([bbox_r[0]+(bbox_r[2]-1)/2,
                                    bbox_r[1]+(bbox_r[3]-1)/2])

        self.center_pos_t = np.array([bbox_t[0]+(bbox_t[2]-1)/2,
                                    bbox_t[1]+(bbox_t[3]-1)/2])

        self.size = np.array([bbox[2], bbox[3]])
        self.rgb_size = np.array([bbox_r[2], bbox_r[3]])
        self.t_size = np.array([bbox_t[2], bbox_t[3]])


        # calculate z crop size
        rgb_w_z = self.rgb_size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.rgb_size)
        rgb_h_z = self.rgb_size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.rgb_size)
        rgb_s_z = round(np.sqrt(rgb_w_z * rgb_h_z))

        t_w_z = self.t_size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.t_size)
        t_h_z = self.t_size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.t_size)
        t_s_z = round(np.sqrt(t_w_z * t_h_z))

        # calculate channle average
        self.channel_average_rgb = np.mean(img_rgb, axis=(0, 1))
        self.channel_average_t = np.mean(img_t, axis=(0, 1))

        # get crop
        z_crop_rgb = self.get_subwindow(img_rgb, self.center_pos_r,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    rgb_s_z, self.channel_average_rgb)
        z_crop_t = self.get_subwindow(img_t, self.center_pos_t,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    t_s_z, self.channel_average_t)
        self.model.template(z_crop_rgb, z_crop_t)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs, hp_score_up_t, p_score_up_t, lrtbs_t):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        score_up_t = self.coarse_location(hp_score_up_t, p_score_up_t, scale_score, lrtbs_t)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        max_r_up_t, max_c_up_t = np.unravel_index(score_up_t.argmax(), score_up_t.shape)

        #
        if score_up[max_r_up][max_c_up] >= score_up_t[max_r_up_t][max_c_up_t]:
            max_r_up = max_r_up
            max_c_up = max_c_up
            c=1
        else:
            max_r_up = max_r_up_t
            max_c_up = max_c_up_t
            c=2

        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy, c

    def track(self, img_rgb, img_t, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)


        x_crop_rgb = self.get_subwindow(img_rgb, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average_rgb)
        x_crop_t = self.get_subwindow(img_t, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average_t)

        outputs = self.model.track(x_crop_rgb,x_crop_t)
        # rgb img pre
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        # t image pre
        cls_t = self._convert_cls(outputs['cls_t']).squeeze()
        cen_t = outputs['cen_t'].data.cpu().numpy()
        cen_t = (cen_t - cen_t.min()) / cen_t.ptp()
        cen_t = cen_t.squeeze()
        lrtbs_t = outputs['loc_t'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        # penalty: s_c, r_c
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        penalty_t = self.cal_penalty(lrtbs_t, hp['penalty_k'])

        p_score = penalty * cls * cen
        p_score_t = penalty_t * cls_t * cen_t

        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
            hp_score_t = p_score_t*(1 - hp['window_lr']) + self.window * hp['window_lr']

        else:
            hp_score = p_score
            hp_score_t = p_score_t

        # feature map resize
        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        hp_score_up_t = cv2.resize(hp_score_t, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up_t = cv2.resize(p_score_t, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up_t = cv2.resize(cls_t, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs_t = np.transpose(lrtbs_t,(1,2,0))
        lrtbs_up_t = cv2.resize(lrtbs_t, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy, c = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs, hp_score_up_t, p_score_up_t, lrtbs_t)
        if c==1:
           hp_score_up = hp_score_up
           p_score_up = p_score_up
           cls_up = cls_up
           lrtbs = lrtbs
           lrtbs_up = lrtbs_up
        else:
           hp_score_up = hp_score_up_t
           p_score_up = p_score_up_t
           cls_up = cls_up_t
           lrtbs = lrtbs_t
           lrtbs_up = lrtbs_up_t
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img_rgb.shape[1])
        cy = bbox_clip(new_cy,0,img_rgb.shape[0])
        width = bbox_clip(new_width,0,img_rgb.shape[1])
        height = bbox_clip(new_height,0,img_rgb.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
               }
