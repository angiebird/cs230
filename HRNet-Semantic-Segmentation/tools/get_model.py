# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config, ext_update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

import run_hrnet as hr


def main():
    cfg = "experiments/bdd100k/bdd100k_resize.yaml"
    model_state_file = "data/pretrained_models/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth"

    logger, final_output_dir, _ = create_logger(
        config, cfg, 'test')

    ext_update_config(config, cfg, [])

    #logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = models.seg_hrnet.get_seg_model(config)

    image_size = config.TEST.IMAGE_SIZE
    dump_input = torch.rand(
        (1, 3, image_size[1], image_size[0])
    )

    #print(get_model_summary(model.cuda(), dump_input.cuda()))


    #logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()

    #torch.from_numpy()
    with torch.no_grad():
        pred = model(dump_input)
        pred = F.upsample(input=pred, size=(
                    image_size[1], image_size[0]), mode='bilinear')
        print(image_size)
        print(pred.shape)

    ## prepare data
    #test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    #test_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                    root=config.DATASET.ROOT,
    #                    list_path=config.DATASET.TEST_SET,
    #                    num_samples=None,
    #                    num_classes=config.DATASET.NUM_CLASSES,
    #                    multi_scale=False,
    #                    flip=False,
    #                    ignore_label=config.TRAIN.IGNORE_LABEL,
    #                    base_size=config.TEST.BASE_SIZE,
    #                    crop_size=test_size,
    #                    downsample_rate=1)

    #testloader = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=1,
    #    shuffle=False,
    #    num_workers=config.WORKERS,
    #    pin_memory=True)
    #
    #start = timeit.default_timer()
    #if 'val' in config.DATASET.TEST_SET:
    #    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
    #                                                       test_dataset, 
    #                                                       testloader, 
    #                                                       model)
    #
    #    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
    #        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
    #        pixel_acc, mean_acc)
    #    logging.info(msg)
    #    logging.info(IoU_array)
    #elif 'test' in config.DATASET.TEST_SET:
    #    test(config, 
    #         test_dataset, 
    #         testloader, 
    #         model,
    #         sv_dir=final_output_dir)

    #end = timeit.default_timer()
    #logger.info('Mins: %d' % np.int((end-start)/60))
    #logger.info('Done')


if __name__ == '__main__':
    main()
