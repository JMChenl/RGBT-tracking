# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.backbone.new_model import resnet50
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise


class GlobalAttentionBlock(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1,bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_feature, t_feature):
        channel_num = rgb_feature.shape[1]
        union_feature = torch.cat((rgb_feature, t_feature), 1)
        b, c, _, _ = union_feature.size()
        # y = self.avg_pool(union_feature).view(b, c)
        y = self.avg_pool(union_feature)
        y = self.fc(y)

        # y = self.fc(y).view(b, c, 1, 1)
        union_feature = union_feature * y.expand_as(union_feature)
        return union_feature[:, :channel_num, :, :], union_feature[:, channel_num:, :, :]


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.rgb_backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        self.t_backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        self.att_rgb_backbone = resnet50([3, 4, 6, 3])
        self.att_t_backbone = resnet50([3, 4, 6, 3])


        # RGBT-Joint_att_template
        self.att_template = GlobalAttentionBlock(512)
        # self.att_template_p2 = GlobalAttentionBlock(1024)
        # self.att_template_p3 = GlobalAttentionBlock(2048)
        # self.att_template_p4 = GlobalAttentionBlock(4096)

        # RGBT-Joint_att_sf
        self.att_xf = CBAM(512)
        # self.att_xf_p2 = CBAM(1024)
        # self.att_xf_p3 = CBAM(2048)
        # self.att_xf_p4 = CBAM(4096)

        # Fusion
        self.rgb_0 = Fusion(1024, 512)
        self.rgb_1 = Fusion(2048, 1024)
        self.rgb_2 = Fusion(4096, 2048)

        self.t_0 = Fusion(1024, 512)
        self.t_1 = Fusion(2048, 1024)
        self.t_2 = Fusion(4096, 2048)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.rgb_neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

            self.t_neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)


        # build car head
        self.car_head = CARHead(cfg, 256)
        self.car_head_t = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        self.down_t = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z_rgb, z_t):
        # get feature
        rgb_zf, p1_rgb_zf = self.rgb_backbone(z_rgb)
        t_zf, p1_t_zf = self.t_backbone(z_t)

        # **** gai_dong ***
        att_rgb_zf ,att_t_zf= self.att_template(p1_rgb_zf, p1_t_zf)

        att_rgb_zf = self.att_rgb_backbone(att_rgb_zf) #  ***gai_dong_l*** att_rgb_zf - > p1_rgb_zf
        att_t_zf = self.att_t_backbone(att_t_zf)

        # att_rgb_zf = self.att_rgb_backbone(p1_rgb_zf)
        # att_t_zf = self.att_t_backbone(p1_t_zf)

        # gain att_rgb_zf, att_t_zf
        # att_rgb_zf[0], att_t_zf[0] = self.att_template_p2(att_rgb_zf[0],att_t_zf[0])
        # att_rgb_zf[1], att_t_zf[1] = self.att_template_p3(att_rgb_zf[1],att_t_zf[1])
        # att_rgb_zf[2], att_t_zf[2] = self.att_template_p4(att_rgb_zf[2],att_t_zf[2])

        # rgb+att_t, t+att_rgb
        rgb_zf[0] = self.rgb_0(rgb_zf[0], att_t_zf[0]) # att_t_zf -> att_rgb_zf
        rgb_zf[1] = self.rgb_1(rgb_zf[1], att_t_zf[1]) # att_t_zf -> att_rgb_zf
        rgb_zf[2] = self.rgb_2(rgb_zf[2], att_t_zf[2]) # att_t_zf -> att_rgb_zf

        t_zf[0] = self.t_0(t_zf[0], att_rgb_zf[0]) # att_rgb_zf -> att_t_zf
        t_zf[1] = self.t_1(t_zf[1], att_rgb_zf[1]) # att_rgb_zf -> att_t_zf
        t_zf[2] = self.t_2(t_zf[2], att_rgb_zf[2]) # att_rgb_zf -> att_t_zf


        if cfg.ADJUST.ADJUST:
            rgb_zf = self.rgb_neck(rgb_zf)
            t_zf = self.t_neck(t_zf)

        self.rgb_zf = rgb_zf
        self.t_zf = t_zf

    def track(self, x_rgb, x_t):
        # get feature
        rgb_xf, p1_rgb_xf = self.rgb_backbone(x_rgb)
        t_xf, p1_t_xf = self.rgb_backbone(x_t)

        union_att_xf = torch.cat([p1_rgb_xf, p1_t_xf], 1) # ***gai_dong_l***
        att_rgb_xf, att_t_xf = self.att_xf(union_att_xf) # ***gai_dong_l***

        att_rgb_xf = self.att_rgb_backbone(att_rgb_xf)  # ***  att_rgb_xf -> p1_rgb_xf
        att_t_xf = self.att_t_backbone(att_t_xf)  # ***  att_t_xf -> p1_t_xf

        # att_rgb_xf = self.att_rgb_backbone(p1_rgb_xf)
        # att_t_xf = self.att_t_backbone(p1_t_xf)

        # gain att_rgb_zf, att_t_zf
        # union_att_xf_p2 = torch.cat([att_rgb_xf[0], att_t_xf[0]], 1)
        # union_att_xf_p3 = torch.cat([att_rgb_xf[1], att_t_xf[1]], 1)
        # union_att_xf_p4 = torch.cat([att_rgb_xf[2], att_t_xf[2]], 1)
        #
        # att_rgb_xf[0], att_t_xf[0] = self.att_xf_p2(union_att_xf_p2)
        # att_rgb_xf[1], att_t_xf[1] = self.att_xf_p3(union_att_xf_p3)
        # att_rgb_xf[2], att_t_xf[2] = self.att_xf_p4(union_att_xf_p4)

        # rgb+att_t, t+att_rgb
        rgb_xf[0] = self.rgb_0(rgb_xf[0], att_t_xf[0]) # att_t_xf -> att_rgb_xf
        rgb_xf[1] = self.rgb_1(rgb_xf[1], att_t_xf[1]) # att_t_zf -> att_rgb_zf
        rgb_xf[2] = self.rgb_2(rgb_xf[2], att_t_xf[2]) # att_t_zf -> att_rgb_zf

        t_xf[0] = self.t_0(t_xf[0], att_rgb_xf[0]) # att_rgb_xf -> att_t_xf
        t_xf[1] = self.t_1(t_xf[1], att_rgb_xf[1]) # att_rgb_xf -> att_t_xf
        t_xf[2] = self.t_2(t_xf[2], att_rgb_xf[2]) # att_rgb_xf -> att_t_xf

        if cfg.ADJUST.ADJUST:
            rgb_xf = self.rgb_neck(rgb_xf)
            t_xf = self.t_neck(t_xf)

        features = self.xcorr_depthwise(rgb_xf[0],self.rgb_zf[0])
        for i in range(len(rgb_xf)-1):
            features_new = self.xcorr_depthwise(rgb_xf[i+1],self.rgb_zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        features_t = self.xcorr_depthwise(t_xf[0], self.t_zf[0])
        for i in range(len(t_xf)-1):
            features_new_t = self.xcorr_depthwise(t_xf[i+1], self.t_zf[i+1])
            features_t = torch.cat([features_t, features_new_t], 1)
        features_t = self.down_t(features_t)

        cls, loc, cen = self.car_head(features)
        cls_t, loc_t, cen_t = self.car_head_t(features_t)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
                'cls_t': cls_t,
                'loc_t': loc_t,
                'cen_t': cen_t
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template_rgb = data['template_rgb'].cuda()
        template_t = data['template_t'].cuda()
        search_rgb = data['search_rgb'].cuda()
        search_t = data['search_t'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        rgb_zf, p1_rgb_zf = self.rgb_backbone(template_rgb)
        t_zf, p1_t_zf = self.t_backbone(template_t)
        rgb_xf, p1_rgb_xf = self.rgb_backbone(search_rgb)
        t_xf, p1_t_xf = self.t_backbone(search_t)

        att_rgb_zf, att_t_zf = self.att_template(p1_rgb_zf, p1_t_zf)
        # ***gai_dong_l***
        union_att_xf = torch.cat([p1_rgb_xf, p1_t_xf],1)
        att_rgb_xf, att_t_xf = self.att_xf(union_att_xf) # ***gai_dong_l***

        att_rgb_zf = self.att_rgb_backbone(att_rgb_zf)
        att_rgb_xf= self.att_rgb_backbone(att_rgb_xf) # ***  att_rgb_xf -> p1_rgb_xf
        att_t_zf = self.att_t_backbone(att_t_zf)
        att_t_xf = self.att_t_backbone(att_t_xf) # *** att_t_xf -> p1_t_xf


        # rgb+att_t add weight
        rgb_zf[0] = self.rgb_0(rgb_zf[0], att_t_zf[0]) # att_t_zf[0] - > att_rgb_zf[0]
        rgb_xf[0] = self.rgb_0(rgb_xf[0], att_t_xf[0]) # att_t_xf[0] - > att_rgb_xf[0]
        t_zf[0] = self.t_0(t_zf[0], att_rgb_zf[0]) # att_rgb_zf[0] - > att_t_zf[0]
        t_xf[0] = self.t_0(t_xf[0], att_rgb_xf[0]) # att_rgb_xf[0] - > att_t_xf[0]

        rgb_zf[1] = self.rgb_1(rgb_zf[1], att_t_zf[1]) # att_t_zf[1] - > att_rgb_zf[1]
        rgb_xf[1] = self.rgb_1(rgb_xf[1], att_t_xf[1]) # att_t_xf[1] - > att_rgb_xf[1]
        t_zf[1] = self.t_1(t_zf[1], att_rgb_zf[1]) # att_rgb_zf[1] - > att_t_zf[1]
        t_xf[1] = self.t_1(t_xf[1], att_rgb_xf[1]) # att_rgb_xf[1] - > att_t_xf[1]

        rgb_zf[2] = self.rgb_2(rgb_zf[2], att_t_zf[2]) # att_t_zf[1] - > att_rgb_zf[1]
        rgb_xf[2] = self.rgb_2(rgb_xf[2], att_t_xf[2]) # att_t_xf[1] - > att_rgb_xf[1]
        t_zf[2] = self.t_2(t_zf[2], att_rgb_zf[2]) # att_rgb_zf[1] - > att_t_zf[1]
        t_xf[2] = self.t_2(t_xf[2], att_rgb_xf[2]) # att_rgb_xf[1] - > att_t_xf[1]


        if cfg.ADJUST.ADJUST:
            rgb_zf = self.rgb_neck(rgb_zf)
            t_zf = self.t_neck(t_zf)
            rgb_xf = self.rgb_neck(rgb_xf)
            t_xf = self.t_neck(t_xf)

        # xf = rgb_xf
        # zf = rgb_zf

        features = self.xcorr_depthwise(rgb_xf[0],rgb_zf[0])
        for i in range(len(rgb_xf)-1):
            features_new = self.xcorr_depthwise(rgb_xf[i+1],rgb_zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        features_t = self.xcorr_depthwise(t_xf[0],t_zf[0])
        for i in range(len(t_xf)-1):
            features_new_t = self.xcorr_depthwise(t_xf[i+1],t_zf[i+1])
            features_t = torch.cat([features_t,features_new_t],1)
        features_t = self.down_t(features_t)


        cls, loc, cen = self.car_head(features)
        cls_t, loc_t, cen_t = self.car_head_t(features_t)

        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        locations_t = compute_locations(cls_t, cfg.TRACK.STRIDE)

        cls = self.log_softmax(cls)
        cls_t = self.log_softmax(cls_t)

        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )
        cls_loss_t, loc_loss_t, cen_loss_t = self.loss_evaluator(
            locations_t,
            cls_t,
            loc_t,
            cen_t, label_cls, label_loc
        )

        # get loss
        outputs = {}
        rgb_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        t_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss_t + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss_t + cfg.TRAIN.CEN_WEIGHT * cen_loss_t
        outputs['total_loss'] = 5*rgb_loss + t_loss
        # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #     cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['rgb_loss'] = rgb_loss
        outputs['t_loss'] = t_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        outputs['cls_loss_t'] = cls_loss_t
        outputs['loc_loss_t'] = loc_loss_t
        outputs['cen_loss_t'] = cen_loss_t

        return outputs




class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            # Flatten(),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels,1, bias=False)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = torch.sigmoid( channel_att_sum )
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate_rgb = SpatialGate()
            self.SpatialGate_t = SpatialGate()
    def forward(self, x):
        channel_num = x.shape[1]//2
        x_out = self.ChannelGate(x)
        x_out_rgb = x_out[:, :channel_num, :, :]
        x_out_t = x_out[:, channel_num:, :, :]
        if not self.no_spatial:
            x_out_rgb = self.SpatialGate_rgb(x_out_rgb)
            x_out_t = self.SpatialGate_t(x_out_t)
        return x_out_rgb, x_out_t

class Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fusion, self).__init__()
        self.fc_down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, future1, future2):
        union_future = torch.cat([future1, future2],1)
        fusion = self.fc_down(union_future)

        return fusion


if __name__ == '__main__':
    model = ModelBuilder()
    data = {}
    data['template'] = torch.FloatTensor(1, 3, 127, 127)
    data['search'] = torch.FloatTensor(1, 3, 225, 225)
    data['label_cls'] = torch.FloatTensor(1, 25, 25)
    data['bbox'] = torch.FloatTensor(1, 4)

    out = model(data)