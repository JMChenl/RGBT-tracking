# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder

from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--cfg', type=str, default='../experiments/siamcar_r50/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param_rgb in getattr(model.rgb_backbone, layer).parameters():
                param_rgb.requires_grad = True
            for param_t in getattr(model.t_backbone, layer).parameters():
                param_t.requires_grad = True
            for m in getattr(model.rgb_backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
            for n in getattr(model.t_backbone, layer).modules():
                if isinstance(n, nn.BatchNorm2d):
                    n.train()

        for att_param_rgb in model.att_rgb_backbone.parameters():
            att_param_rgb.requires_grad = True
        for att1 in model.att_rgb_backbone.modules():
            if isinstance(att1, nn.BatchNorm2d):
                att1.train()
        for att_param_t in model.att_t_backbone.parameters():
            att_param_t.requires_grad = True
        for att2 in model.att_t_backbone.modules():
            if isinstance(att2, nn.BatchNorm2d):
                att2.train()

    else:
        for param_rgb in model.rgb_backbone.parameters():
            param_rgb.requires_grad = False
        for param_t in model.t_backbone.parameters():
            param_t.requires_grad = False
        for m in model.rgb_backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for n in model.t_backbone.modules():
            if isinstance(n, nn.BatchNorm2d):
                n.eval()
        for att_param_rgb in model.att_rgb_backbone.parameters():
            att_param_rgb.requires_grad = False
        for att1 in model.att_rgb_backbone.modules():
            if isinstance(att1, nn.BatchNorm2d):
                att1.eval()
        for att_param_t in model.att_t_backbone.parameters():
            att_param_t.requires_grad = False
        for att2 in model.att_t_backbone.modules():
            if isinstance(att2, nn.BatchNorm2d):
                att2.eval()

    for k, v in model.rgb_backbone.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    trainable_params = []
    # trainable_params_t = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.rgb_backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.t_backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR*10}]

    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.att_rgb_backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.att_t_backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR*10}]

    trainable_params += [{'params': model.att_template.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    # trainable_params += [{'params': model.att_template.parameters(),
    #                       'lr': cfg.TRAIN.BASE_LR}]
    # trainable_params += [{'params': model.att_template.parameters(),
    #                       'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.att_xf.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    # trainable_params += [{'params': model.att_xf.parameters(),
    #                       'lr': cfg.TRAIN.BASE_LR}]
    # trainable_params += [{'params': model.att_xf.parameters(),
    #                       'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rgb_0.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.rgb_1.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.rgb_2.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.t_0.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.t_1.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.t_2.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.rgb_neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.t_neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]


    trainable_params += [{'params': model.car_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.car_head_t.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.down.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.down_t.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, car_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            car_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + car_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    car_norm = car_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/car', car_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)
        loss = outputs['total_loss'].mean()

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.mean().data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    # rank = 0
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    device = torch.device("cuda:1")
    model = ModelBuilder().train()
    dist_model = nn.DataParallel(model).to(device)#.cuda()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, 'pretrained_models', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.rgb_backbone, backbone_path)
        load_pretrain(model.t_backbone, backbone_path)
        load_pretrain(model.att_rgb_backbone, backbone_path)
        load_pretrain(model.att_t_backbone, backbone_path)


    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = nn.DataParallel(model,device_ids=[1,2,4,5])

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
