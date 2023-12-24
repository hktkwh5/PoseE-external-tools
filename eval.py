import os
import csv
import torch
import numpy as np
import argparse
from lib.dataset import MPII3D
from lib.models import VIBE,VIBE_cross_eye
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader
from lib.core.loss import VIBELoss_cross_eye
from lib.models import MotionDiscriminator_cross_eye
from lib.utils.utils import get_optimizer

def main(cfg,args):
    print('...Evaluating on 3DPW test set...')

    model = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    # model = VIBE(
    #     n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
    #     batch_size=cfg.TRAIN.BATCH_SIZE,
    #     seqlen=cfg.DATASET.SEQLEN,
    #     hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
    #     pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
    #     add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
    #     bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
    #     use_residual=cfg.MODEL.TGRU.RESIDUAL,
    # ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()
    file_name='./lib/data/video_dir/iris_test1_MPI_S7.csv'#I:\\diskF\\MPI-INF-3DHP(beifen)\\video_dir\\iris_test1_MPI_S7.csv #I:\\diskF\\H36M\\video_dir\\iris_training2_H36M_S9_11.csv
    f1 = open(file_name, 'r')
    f2 = open(file_name, 'r')
    f3 = open(file_name, 'r')
    data_label3 = list(csv.reader(f3))[1:]
    data_label3 = list(map(lambda x: x[1:], data_label3))
    data_label3 = np.array(data_label3).astype(dtype=int)

    data_label3 = list(data_label3)
    data_label1 = list(csv.reader(f1))[1:]
    data_label1 = list(map(lambda x: x[1:], data_label1))
    data_label2 = list(csv.reader(f2))[1:]
    data_label2 = list(map(lambda x: x[1:], data_label2))
    data_label1 = np.array(data_label1).astype(dtype=int)
    data_label2 = np.array(data_label2).astype(dtype=int)
    data_label1 = list(data_label1)
    data_label2 = list(data_label2)
    for index in range(len(data_label2)):
        con_1=[]
        con_2=[]
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
    test_db = MPII3D(set='val', seqlen=cfg.DATASET.SEQLEN,iris_training1=data_label1,
            iris_training2=data_label2,iris_training3=data_label3, debug=cfg.DEBUG)

    data_loaders = None
    generator_cross_eye = None
    loss_cross_eye = None
    gen_optimizer = None
    motion_discriminator_cross_eye = None
    dis_motion_optimizer_cross_eye = None
    motion_lr_scheduler_cross_eye = None
    lr_scheduler_cross_eye = None
    if(args.istrain):
        generator_cross_eye = VIBE_cross_eye(
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seqlen=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)
        loss_cross_eye = VIBELoss_cross_eye(
            e_loss_weight=cfg.LOSS.KP_2D_W,
            e_3d_loss_weight=cfg.LOSS.KP_3D_W,
            e_pose_loss_weight=cfg.LOSS.POSE_W,
            e_shape_loss_weight=cfg.LOSS.SHAPE_W,
            d_motion_loss_weight=cfg.LOSS.D_MOTION_LOSS_W,
        )
        gen_optimizer = get_optimizer(
            model=generator_cross_eye,
            optim_type=cfg.TRAIN.GEN_OPTIM,
            lr=cfg.TRAIN.GEN_LR,
            weight_decay=cfg.TRAIN.GEN_WD,
            momentum=cfg.TRAIN.GEN_MOMENTUM,
        )

        motion_discriminator_cross_eye = MotionDiscriminator_cross_eye(
            rnn_size=cfg.TRAIN.MOT_DISCR.HIDDEN_SIZE,
            input_size=69,
            num_layers=cfg.TRAIN.MOT_DISCR.NUM_LAYERS,
            output_size=1,
            feature_pool=cfg.TRAIN.MOT_DISCR.FEATURE_POOL,
            attention_size=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL != 'attention' else cfg.TRAIN.MOT_DISCR.ATT.SIZE,
            attention_layers=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL != 'attention' else cfg.TRAIN.MOT_DISCR.ATT.LAYERS,
            attention_dropout=None if cfg.TRAIN.MOT_DISCR.FEATURE_POOL != 'attention' else cfg.TRAIN.MOT_DISCR.ATT.DROPOUT
        ).to(cfg.DEVICE)

        dis_motion_optimizer_cross_eye = get_optimizer(
            model=motion_discriminator_cross_eye,
            optim_type=cfg.TRAIN.MOT_DISCR.OPTIM,
            lr=cfg.TRAIN.MOT_DISCR.LR,
            weight_decay=cfg.TRAIN.MOT_DISCR.WD,
            momentum=cfg.TRAIN.MOT_DISCR.MOMENTUM
        )

        motion_lr_scheduler_cross_eye = torch.optim.lr_scheduler.ReduceLROnPlateau(
            dis_motion_optimizer_cross_eye,
            mode='min',
            factor=0.1,
            patience=cfg.TRAIN.LR_PATIENCE,
            verbose=True,
        )

        lr_scheduler_cross_eye = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gen_optimizer,
            mode='min',
            factor=0.1,
            patience=cfg.TRAIN.LR_PATIENCE,
            verbose=True,
        )



    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS)



    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
        istrain=args.istrain,
        data_loaders=data_loaders,

        generator=generator_cross_eye,
        motion_discriminator=motion_discriminator_cross_eye,
        criterion=loss_cross_eye,
        dis_motion_optimizer=dis_motion_optimizer_cross_eye,
        dis_motion_update_steps=cfg.TRAIN.MOT_DISCR.UPDATE_STEPS,
        gen_optimizer=gen_optimizer,
        lr_scheduler=lr_scheduler_cross_eye,
        motion_lr_scheduler=motion_lr_scheduler_cross_eye,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
    ).run(cfg,args)

if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--tracker_batch_size', type=int, default=12,#12
                        help='batch size of object detector used for bbox tracking')
    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')
    parser.add_argument('--vibe_batch_size', type=int, default=450,#450
                        help='batch size of VIBE')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')
    parser.add_argument('--no_render', action='store_true', default=False,
                        help='disable final rendering of output video.')
    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')
    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')
    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')
    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')
    parser.add_argument('--istrain', type=bool, default=True)
    args = parser.parse_args()
    main(cfg,args)
