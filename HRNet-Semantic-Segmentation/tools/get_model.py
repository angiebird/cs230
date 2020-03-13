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
import matplotlib.image as mpimg
from PIL import Image


def get_video_image(seg_hash, time_idx = 1000):
    base_dir = "data/cityscapes/video_images/train/resize/"
    img_file = os.path.join(base_dir, seg_hash + "_"+ str(time_idx) + ".png")
    image =  mpimg.imread(img_file)
    image = np.moveaxis(image, 2, 0)
    image = image.reshape([1] + list(image.shape))
    image = torch.from_numpy(image)
    return image

def get_label(one_hot):
    one_hot = np.array(one_hot)
    label = np.argmax(one_hot, axis = 0)                                                                                                
    label = label.astype("uint8")
    return label

def remove_last_layer(model):
    model.last_layer = nn.Sequential(*list(model.last_layer[:-1]))
    return model


def gen_feature(model, image_size, seg_hash, time_idx):
    image = get_video_image(seg_hash, time_idx = time_idx)
    pred = model(image)
    pred = F.upsample(input=pred, size=(
                image_size[1], image_size[0]), mode='bilinear')
    pred = np.array(pred.cpu())
    return pred[0]

def gen_feature_list(model, image_size, seg_hash_list):
    feature_dir = "feature_data"
    for seg_hash in seg_hash_list:
        index_list = range(975, 1005, 5)
        for idx in index_list:
            feature = gen_feature(model, image_size, seg_hash, idx)
            file_path = os.path.join(feature_dir, seg_hash + "_" + str(idx))
            np.save(file_path, feature)

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

    with torch.no_grad():
        seg_hash_list = hr.get_train_list("../seg_hash_list.txt")
        gen_feature_list(model, image_size, seg_hash_list)

if __name__ == '__main__':
    main()
