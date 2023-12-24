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
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import csv
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.loss import VIBELoss
from lib.dataset import MPII3D
from lib.core.trainer import Trainer
from torch.utils.data import DataLoader
from lib.core.config import parse_args
from lib.utils.utils import prepare_output_dir
from lib.models import VIBE, MotionDiscriminator
from lib.dataset.loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= Dataloaders ========= #
    # data_loaders = get_data_loaders(cfg)

    file_name = 'F:\\MPI-INF-3DHP(beifen)\\video_dir\\iris_test1_MPI_S678.csv'
    f1 = open(file_name,
              'r')  # 这里的是关于关节超界的算法，根据出界的情况，该编码长度为5位     C:\\MPI-INF-3DHP(beifen)\\video_dir\\iris_training1_MPI_S8.csv   F:\\MPI-INF-3DHP(beifen)\\video_dir\\iris_test1_MPI_S7.csv    iris_valid1_MPI_S12345.csv
    f2 = open(file_name, 'r')  # 这里的是关于人的伸展程度的编码，该编码为8位F:\\H36M\\video_dir\\iris_training2_H36M_S9_11.csv
    f3 = open(file_name, 'r')
    data_label3 = list(csv.reader(f3))[1:]
    data_label3 = list(map(lambda x: x[1:], data_label3))
    data_label3 = np.array(data_label3).astype(dtype=int)

    data_label3 = list(data_label3)
    data_label1 = list(csv.reader(f1))[1:]
    data_label1 = list(map(lambda x: x[1:], data_label1))
    data_label2 = list(csv.reader(f2))[1:]
    data_label2 = list(map(lambda x: x[1:], data_label2))
    data_label1 = np.array(data_label1).astype(dtype=int)  # 这里的是关于关节超界的算法，根据出界的情况，该编码长度为5位
    data_label2 = np.array(data_label2).astype(dtype=int)
    data_label1 = list(data_label1)
    data_label2 = list(data_label2)
    for index in range(len(data_label2)):
        con_1 = []
        con_2 = []
        con_3 = []
        for index1 in range(len(data_label1[index])):
            con_1.append(int(data_label1[index][index1]))
        for index2 in range(len(data_label2[index])):
            con_2.append(int(data_label2[index][index2]))
        for index3 in range(len(data_label3[index])):
            con_3.append(int(data_label3[index][index3]))
        data_label1[index] = con_1
        data_label2[index] = con_2
        data_label3[index] = con_3
    test_db = MPII3D(set='val', seqlen=cfg.DATASET.SEQLEN, iris_training1=data_label1,
                     iris_training2=data_label2, iris_training3=data_label3, debug=cfg.DEBUG)
    train_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,  # 这是32#cfg.TRAIN.BATCH_SIZE
        shuffle=False,
        num_workers=cfg.NUM_WORKERS)

    # ========= Compile Loss ========= #
    loss = VIBELoss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
        d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W,
    )

    # ========= Initialize networks, optimizers and lr_schedulers ========= #
    generator = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        generator.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')

    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    motion_discriminator = MotionDiscriminator(
        rnn_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
        input_size=69,
        num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
        output_size=1,
        feature_pool=cfg.TRAIN.MOT_DISCR.FEATURE_POOL,
        attention_size=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.SIZE,
        attention_layers=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.LAYERS,
        attention_dropout=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL !='attention' else cfg.TRAIN.MOT_DISCR.ATT.DROPOUT
    ).to(cfg.DEVICE)

    dis_motion_optimizer = get_optimizer(
        model=motion_discriminator,
        optim_type=cfg.TRAIN.MOT_DISCR.OPTIM,
        lr=cfg.TRAIN.MOT_DISCR.LR,
        weight_decay=cfg.TRAIN.MOT_DISCR.WD,
        momentum=cfg.TRAIN.MOT_DISCR.MOMENTUM
    )

    motion_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dis_motion_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )

    # ========= Start Training ========= #
    Trainer(
        data_loaders=train_loader,
        generator=generator,
        motion_discriminator=motion_discriminator,
        criterion=loss,
        dis_motion_optimizer=dis_motion_optimizer,
        dis_motion_update_steps=cfg.TRAIN.MOT_DISCR.UPDATE_STEPS,
        gen_optimizer=gen_optimizer,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        logdir=cfg.LOGDIR,
        lr_scheduler=lr_scheduler,
        motion_lr_scheduler=motion_lr_scheduler,
        resume=cfg.TRAIN.RESUME,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
        debug_freq=cfg.DEBUG_FREQ,
    ).fit()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
