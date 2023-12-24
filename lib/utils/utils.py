# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import yaml
import time
import torch
import numpy as np
import cv2
import shutil
import logging
import operator
from tqdm import tqdm
from os import path as osp
from functools import reduce
from typing import List, Union


def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)


def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def iterdict(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = dict(v)
            iterdict(v)
    return d


def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def step_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def read_yaml(filename):
    return yaml.load(open(filename, 'r'))


def write_yaml(filename, object):
    with open(filename, 'w') as f:
        yaml.dump(object, f)


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def save_to_file(obj, filename, mode='w'):
    with open(filename, mode) as f:
        f.write(obj)


def concatenate_dicts(dict_list, dim=0):
    rdict = dict.fromkeys(dict_list[0].keys())
    for k in rdict.keys():
        rdict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return rdict


def bool_to_string(x: Union[List[bool],bool]) ->  Union[List[str],str]:
    """
    boolean to string conversion
    :param x: list or bool to be converted
    :return: string converted thing
    """
    if isinstance(x, bool):
        return [str(x)]
    for i, j in enumerate(x):
        x[i]=str(j)
    return x


def checkpoint2model(checkpoint, key='gen_state_dict'):
    state_dict = checkpoint[key]
    print(f'Performance of loaded model on 3DPW is {checkpoint["performance"]:.2f}mm')
    # del state_dict['regressor.mean_theta']
    return state_dict


def get_optimizer(model, optim_type, lr, weight_decay, momentum):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        raise ModuleNotFoundError
    return opt


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg

def get_3rd_point(a, b):#中心点减去中上角点的坐标
    direct = a - b#direct的第一项为0，第二项为中心点的y坐标
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)#[-y, 0] 所以返回的是[0,0]
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)#sn, cs =0,1

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                        scale_tmp=200.0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * scale_tmp#scale是人物所在方框的宽和高，但是除了200
    src_w = scale_tmp[0]#原始坐标框的宽
    dst_w = output_size[0]#转换后坐标的宽
    dst_h = output_size[1]#转换后坐标的高

    rot_rad = np.pi * rot / 180#rot是0
    src_dir = get_dir([0, src_w * -0.5], rot_rad)#由于rot_rad为0，所以输出就是输入的-0.5倍。即：
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]#这里应该是转换后的中心点
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir#输出像素的中上角的点
    #    src：原始图像中的三个点的坐标
    # dst：变换后的这三个点对应的坐标
    # M：根据三个对应点求出的仿射变换矩阵
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):#关节点坐标，初始。t为转换系数
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]