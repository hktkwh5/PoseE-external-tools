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

import time
import os
import pickle
from lib.models import spin
import logging
import os.path as osp
import random
from progress.bar import Bar
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.RPSM_utils import *
from lib.core.correct_matrix_math import Curr_matrix_toTar
from lib.core.Conv_BatchSeqImage_s import Conv_BatchSeqImage_sInterpolation
from lib.data_utils.img_utils import normalize_2d_kp, transfrom_keypoints
from lib.core.config import VIBE_DATA_DIR
from lib.utils.utils import move_dict_to_device
from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    batch_compute_similarity_transform_torch,
)
logger = logging.getLogger(__name__)

def get_single_item(db,index,batch_size):
    target_batch={}
    for batch_index in range(batch_size):
        if((index+batch_index)<len(db.dataset.vid_indices)):
            start_index, end_index = db.dataset.vid_indices[index+batch_index]
        else:
            continue
        is_train = db.dataset.set == 'train'

        if db.dataset.dataset_name == '3dpw':
            kp_2d = convert_kps(db.dataset.db['joints2D'][start_index:end_index + 1], src='common', dst='spin')
            kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
        elif db.dataset.dataset_name == 'mpii3d':
            kp_2d = db.dataset.db['joints2D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        elif db.dataset.dataset_name == 'h36m':
            kp_2d = db.dataset.db['joints2D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        kp_2d_tensor = np.ones((db.dataset.seqlen, 49, 3), dtype=np.float16)
        nj = 14 if not is_train else 49 #原本是nj = 14 if not is_train else 49
        kp_3d_tensor = np.zeros((db.dataset.seqlen, nj, 3), dtype=np.float16)

        if db.dataset.dataset_name == '3dpw':
            pose = db.dataset.db['pose'][start_index:end_index + 1]
            shape = db.dataset.db['shape'][start_index:end_index + 1]
            w_smpl = torch.ones(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(db.seqlen).float()
                w_3d = torch.ones(db.seqlen).float()
            else:
                pose = db.dataset.db['pose'][start_index:end_index + 1]
                shape = db.dataset.db['shape'][start_index:end_index + 1]
                w_smpl = torch.ones(db.dataset.seqlen).float()
                w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'mpii3d':
            pose = np.zeros((kp_2d.shape[0], 72))  # pose = np.zeros((kp_3d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))  # shape = np.zeros((kp_3d.shape[0], 10))
            w_smpl = torch.zeros(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()

        bbox = db.dataset.db['bbox'][start_index:end_index + 1]
        input = torch.from_numpy(db.dataset.db['features'][start_index:end_index + 1]).float()
        theta_tensor = np.zeros((db.dataset.seqlen, 85), dtype=np.float16)
        for idx in range(db.dataset.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx, :, :2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx, :, :2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)
            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            'features': input,
            'theta': torch.from_numpy(theta_tensor).float(),  # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),  # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),  # 3D keypoints
            'w_smpl': w_smpl,
            'w_3d': w_3d,
            'bbox': torch.from_numpy(bbox),
            'image_name': image_name
        }

        if db.dataset.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = db.dataset.db['valid_i'][start_index:end_index + 1]

        if db.dataset.dataset_name == '3dpw' and not is_train:
            vn = db.dataset.db['vid_name'][start_index:end_index + 1]
            fi = db.dataset.db['frame_id'][start_index:end_index + 1]
            target['instance_id'] = [f'{v}/{f}' for v, f in zip(vn, fi)]
        if db.dataset.debug:
            from lib.data_utils.img_utils import get_single_image_crop
            if db.dataset.dataset_name == 'mpii3d':
                video = db.dataset.db['img_name'][start_index:end_index + 1]
            elif db.dataset.dataset_name == 'h36m':
                video = db.dataset.db['img_name'][start_index:end_index + 1]
            else:
                vid_name = db.dataset.db['vid_name'][start_index]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                f = osp.join(db.dataset.folder, 'imageFiles', vid_name)
                video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                frame_idxs = db.dataset.db['frame_id'][start_index:end_index + 1]
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video
        target.pop('valid')
        if (batch_index == 0):
            for key,value in target.items():
                shape=[1]+list(value.shape)
                if(key=='image_name'):
                    target_batch[key]=value.reshape(shape)
                else:
                    target_batch[key] = value.view(shape)
        else:
            for key, value in target.items():
                if (key == 'image_name'):
                    target_batch[key] = np.concatenate((target_batch[key], value.reshape([1] + list(value.shape))), axis=0)
                else:
                    target_batch[key] = torch.cat((target_batch[key],value.view([1]+list(value.shape))), dim=0)
    return target_batch

def get_single_item_genert_ground(db,batch_size_list):
    target_batch = {}
    for batch_index in batch_size_list:
        if ((batch_index) < len(db.dataset.vid_indices)):
            start_index, end_index = db.dataset.vid_indices[batch_index]
        else:
            continue
        is_train = db.dataset.set == 'train'

        if db.dataset.dataset_name == '3dpw':
            kp_2d = convert_kps(db.dataset.db['joints2D'][start_index:end_index + 1], src='common', dst='spin')
            kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
        elif db.dataset.dataset_name == 'mpii3d':
            kp_2d = db.dataset.db['joints2D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        elif db.dataset.dataset_name == 'h36m':
            kp_2d = db.dataset.db['joints2D'][start_index:end_index + 1]
            image_name = db.dataset.db['img_name'][start_index:end_index + 1]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        kp_2d_tensor = np.ones((db.dataset.seqlen, 49, 3), dtype=np.float16)
        nj = 14 if not is_train else 49
        kp_3d_tensor = np.zeros((db.dataset.seqlen, nj, 3), dtype=np.float16)

        if db.dataset.dataset_name == '3dpw':
            pose = db.dataset.db['pose'][start_index:end_index + 1]
            shape = db.dataset.db['shape'][start_index:end_index + 1]
            w_smpl = torch.ones(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(db.seqlen).float()
                w_3d = torch.ones(db.seqlen).float()
            else:
                pose = db.dataset.db['pose'][start_index:end_index + 1]
                shape = db.dataset.db['shape'][start_index:end_index + 1]
                w_smpl = torch.ones(db.dataset.seqlen).float()
                w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'mpii3d':
            pose = np.zeros((kp_2d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))
            w_smpl = torch.zeros(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()

        bbox = db.dataset.db['bbox'][start_index:end_index + 1]
        input = torch.from_numpy(db.dataset.db['features'][start_index:end_index + 1]).float()
        theta_tensor = np.zeros((db.dataset.seqlen, 85), dtype=np.float16)
        for idx in range(db.dataset.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx, :, :2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx, :, :2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)
            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            'features': input,
            'theta': torch.from_numpy(theta_tensor).float(),  # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),  # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),  # 3D keypoints
            'w_smpl': w_smpl,
            'w_3d': w_3d,
            'bbox': torch.from_numpy(bbox),
            'image_name': image_name
        }

        if db.dataset.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = db.dataset.db['valid_i'][start_index:end_index + 1]

        if db.dataset.dataset_name == '3dpw' and not is_train:
            vn = db.dataset.db['vid_name'][start_index:end_index + 1]
            fi = db.dataset.db['frame_id'][start_index:end_index + 1]
            target['instance_id'] = [f'{v}/{f}' for v, f in zip(vn, fi)]
        if db.dataset.debug:
            from lib.data_utils.img_utils import get_single_image_crop
            if db.dataset.dataset_name == 'mpii3d':
                video = db.dataset.db['img_name'][start_index:end_index + 1]
                # print(video)
            elif db.dataset.dataset_name == 'h36m':
                video = db.dataset.db['img_name'][start_index:end_index + 1]
            else:
                vid_name = db.dataset.db['vid_name'][start_index:end_index + 1]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                f = osp.join(db.dataset.folder, 'imageFiles', vid_name)
                video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                frame_idxs = db.dataset.db['frame_id'][start_index:end_index + 1]
                # print(f, frame_idxs)
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video
        target.pop('valid')
        if (batch_index == batch_size_list[0]):
            for key,value in target.items():
                shape=[1]+list(value.shape)
                if(key=='image_name'):
                    target_batch[key]=value.reshape(shape)
                else:
                    target_batch[key] = value.view(shape)
        else:
            for key, value in target.items():
                if (key == 'image_name'):
                    target_batch[key] = np.concatenate((target_batch[key], value.reshape([1] + list(value.shape))), axis=0)
                else:
                    target_batch[key] = torch.cat((target_batch[key],value.view([1]+list(value.shape))), dim=0)
    return target_batch

def get_single_item_genert_ground_2(db,batch_size_list):
    target_batch = {}
    for batch_index in range(0, len(batch_size_list), 16):
        start_index = batch_size_list[batch_index:batch_index + 16]
        is_train = db.dataset.set == 'train'

        if db.dataset.dataset_name == '3dpw':
            kp_2d = convert_kps(db.dataset.db['joints2D'][start_index], src='common', dst='spin')
            kp_3d = db.dataset.db['joints3D'][start_index]
            image_name = db.dataset.db['img_name'][start_index]
        elif db.dataset.dataset_name == 'mpii3d':
            kp_2d = db.dataset.db['joints2D'][start_index]
            image_name = db.dataset.db['img_name'][start_index]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index], src='spin', dst='common')
        elif db.dataset.dataset_name == 'h36m':
            kp_2d = db.dataset.db['joints2D'][start_index]
            image_name = db.dataset.db['img_name'][start_index]
            if is_train:
                kp_3d = db.dataset.db['joints3D'][start_index]
            else:
                kp_3d = convert_kps(db.dataset.db['joints3D'][start_index], src='spin', dst='common')
        kp_2d_tensor = np.ones((db.dataset.seqlen, 49, 3), dtype=np.float16)
        nj = 14 if not is_train else 49 #原本是nj = 14 if not is_train else 49
        kp_3d_tensor = np.zeros((db.dataset.seqlen, nj, 3), dtype=np.float16)

        if db.dataset.dataset_name == '3dpw':
            pose = db.dataset.db['pose'][start_index]
            shape = db.dataset.db['shape'][start_index]
            w_smpl = torch.ones(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(db.seqlen).float()
                w_3d = torch.ones(db.seqlen).float()
            else:
                pose = db.dataset.db['pose'][start_index]
                shape = db.dataset.db['shape'][start_index]
                w_smpl = torch.ones(db.dataset.seqlen).float()
                w_3d = torch.ones(db.dataset.seqlen).float()
        elif db.dataset.dataset_name == 'mpii3d':
            pose = np.zeros((kp_2d.shape[0], 72))  # pose = np.zeros((kp_3d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))  # shape = np.zeros((kp_3d.shape[0], 10))
            w_smpl = torch.zeros(db.dataset.seqlen).float()
            w_3d = torch.ones(db.dataset.seqlen).float()

        bbox = db.dataset.db['bbox'][start_index]
        input = torch.from_numpy(db.dataset.db['features'][start_index]).float()
        theta_tensor = np.zeros((db.dataset.seqlen, 85), dtype=np.float16)
        for idx in range(db.dataset.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx, :, :2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx, :, :2],  # kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)
            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            'features': input,
            'theta': torch.from_numpy(theta_tensor).float(),  # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),  # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),  # 3D keypoints
            'w_smpl': w_smpl,
            'w_3d': w_3d,
            'bbox': torch.from_numpy(bbox),
            'image_name': image_name
        }

        if db.dataset.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = db.dataset.db['valid_i'][start_index]

        if db.dataset.dataset_name == '3dpw' and not is_train:
            vn = db.dataset.db['vid_name'][start_index]
            fi = db.dataset.db['frame_id'][start_index]
            target['instance_id'] = [f'{v}/{f}' for v, f in zip(vn, fi)]
        if db.dataset.debug:
            from lib.data_utils.img_utils import get_single_image_crop
            if db.dataset.dataset_name == 'mpii3d':
                video = db.dataset.db['img_name'][start_index]
                # print(video)
            elif db.dataset.dataset_name == 'h36m':
                video = db.dataset.db['img_name'][start_index]
            else:
                vid_name = db.dataset.db['vid_name'][start_index]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                f = osp.join(db.dataset.folder, 'imageFiles', vid_name)
                video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                frame_idxs = db.dataset.db['frame_id'][start_index]
                # print(f, frame_idxs)
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video
        target.pop('valid')
        if (batch_size_list[batch_index] == batch_size_list[0]):
            for key,value in target.items():
                shape=[1]+list(value.shape)
                if(key=='image_name'):
                    target_batch[key]=value.reshape(shape)
                else:
                    target_batch[key] = value.view(shape)
        else:
            for key, value in target.items():
                if (key == 'image_name'):
                    target_batch[key] = np.concatenate((target_batch[key], value.reshape([1] + list(value.shape))), axis=0)
                else:
                    target_batch[key] = torch.cat((target_batch[key],value.view([1]+list(value.shape))), dim=0)
    return target_batch

class Evaluator():
    def __init__(
            self,
            test_loader,
            model,
            device=None,
            istrain=False,
            data_loaders=None,

            generator=None,
            motion_discriminator=None,
            criterion=None,
            dis_motion_optimizer=None,
            dis_motion_update_steps=None,
            gen_optimizer=None,
            lr_scheduler=None,
            motion_lr_scheduler=None,
            num_iters_per_epoch = 1000,
    ):
        self.test_loader = test_loader
        self.train_global_step = 0
        self.valid_global_step = 0
        self.disc_motion_loader = data_loaders

        self.generator = generator
        self.motion_discriminator = motion_discriminator
        self.criterion = criterion
        self.dis_motion_optimizer = dis_motion_optimizer
        self.dis_motion_update_steps = dis_motion_update_steps
        self.gen_optimizer = gen_optimizer
        self.lr_scheduler = lr_scheduler
        self.motion_lr_scheduler = motion_lr_scheduler
        self.num_iters_per_epoch = num_iters_per_epoch

        self.end_index=self.test_loader.dataset.vid_indices[-1][-1]
        self.frame_ids_al=self.test_loader.dataset.db['frame_id']
        self.avali_frame =self.test_loader.dataset.vid_indices
        self.model = model
        self.device = device
        self.istrain = istrain

        self.evaluation_accumulators0 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators1 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators2 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators3 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators4 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators_just = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators_just0 = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d','camera_para','center','scaler'])
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.spin_model = spin.get_pretrained_hmr()

    def validate(self,args,weight,Scale_testS,testSamNum):#Scale_testS
        self.model.eval()
        start = time.time()
        bar = Bar('Validation', fill='#', max=len(self.test_loader))
        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()#
        camera_list,center_list,scale_list=[],[],[]
        justice_Sx = 'S0'
        OrigMP_0_Sum,OrigMP_1_Sum,Curr_MP_Sum,Orig_PaMP0_Sum,Orig_PaMP1_Sum,Curr_PaMP_Sum =0,0,0,0,0,0
        OrigMP_1M_Sum, Curr_MP_MSum, Orig_PaMP1_MSum, Curr_PaMP_MSum = 0,0,0,0
        Joint_axis_Opti = torch.zeros((1, 14, 3))
        Number_MP=0
        Joint_axis_Filter = np.array(
            [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]),
             np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]),
             np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]),
             np.array([1, 1, 1]), np.array([1, 1, 1])])

        for i in range(0,len(self.test_loader.dataset.vid_indices)):
            target = get_single_item(self.test_loader,i,self.test_loader.batch_size)

            # file_name = 'I:\\diskD\\Icon_out_data\\total_H36M_info.pkl'
            # f = open(file_name, 'wb')
            # info = {'feature': [self.test_loader.dataset.db['features']],
            #         'image_name': [self.test_loader.dataset.db['img_name']],
            #         'target_pose': [self.test_loader.dataset.db['joints3D']],
            #         'dir_name': [self.test_loader.dataset.db['valid_i']],
            #         'camera': [self.test_loader.dataset.camera_list],
            #         'bbox': [self.test_loader.dataset.db['bbox']], '2d_pose': [self.test_loader.dataset.db['joints2D']]}
            # # print(self.test_loader.dataset.camera_list[6000:6010])
            # pickle.dump(info, f)
            # f.close()
            move_dict_to_device(target, self.device)
            # <=============
            if(len(target['features'])!=self.test_loader.batch_size):
                break

            with torch.no_grad():
                inp = target['features']
                image_name = target['image_name']
                self.real_motion_samples = target


                if (i == 0 or self.test_loader.dataset.db['frame_id'][self.test_loader.dataset.vid_indices[i][0]][:2] != justice_Sx):
                    justice_Sx = self.test_loader.dataset.db['frame_id'][self.test_loader.dataset.vid_indices[i][0]][:2]
                    Batch_index_List=[]
                    for index in range(self.test_loader.batch_size*len(target['features'][0])):
                        Batch_index_List.append(random.randint(0, len(self.test_loader.dataset.vid_indices)-1))
                    Tar_Gengro = get_single_item_genert_ground_2(self.test_loader, Batch_index_List)
                    Tra_Gro3Dtar = Tar_Gengro['kp_3d']
                    Tra_GroSamImg = Tar_Gengro['image_name']
                    continue

                Orig_MP0, Orig_PaMP0 = 0, 0
                Orig_MP1, Orig_PaMP1  = 0, 0
                Curr_MP, Curr_PaMP = 0, 0

                Orig_MP1_mMP, Orig_PaMP1_mMP = 0, 0
                Curr_MP_mMP, Curr_PaMP_mMP = 0, 0
                Batch_Seq_Num=0
                for index1 in range(1, len(inp),10):#for index1 in range(1, len(inp),10)  #for index1 in range(1, len(inp),6):
                    for index2 in range(0,len(inp[0]),3): #for index2 in range(0,len(inp[0]),3):   #for index2 in range(len(inp[0])):
                        Batch_Seq_Num+=1
                        batch, seq = index1, index2

                        inp_trans, retuen_target_3D, Batch_index_List,seq_index_list,count_index_list,center_batch, center_seq, origin_count = Conv_BatchSeqImage_sInterpolation(
                            self.spin_model, inp, batch, seq, target['kp_3d'],image_name,Tra_Gro3Dtar,Tra_GroSamImg,Scale_testS,testSamNum)
                        Preds = self.model(inp, J_regressor=J_regressor)
                        Preds_Trans = self.model(inp_trans, J_regressor=J_regressor)
                        Tar_Pelvis = (target['kp_3d'][batch][seq][[2], :] + target['kp_3d'][batch][seq][[3], :]) / 2.0

                        Target_Con = target['kp_3d'][batch][seq] - Tar_Pelvis
                        Pred_PelS = (Preds[0]['kp_3d'][batch][seq][[2], :] + Preds[0]['kp_3d'][batch][seq][[3], :]) / 2.0
                        Preds_ConS = Preds[0]['kp_3d'][batch][seq] - Pred_PelS
                        Errors_Singe = torch.sqrt(((Preds_ConS - Target_Con) ** 2).sum(dim=-1)).mean(
                            dim=-1).cpu().numpy()
                        MP_Singe = Errors_Singe * 1000
                        Orig_MP0 += MP_Singe
                        Preds_ConS = Preds_ConS.reshape(1,Preds_ConS.size(0),Preds_ConS.size(1))
                        Target_Con = Target_Con.reshape(1,Target_Con.size(0),Target_Con.size(1))
                        S1_hat = batch_compute_similarity_transform_torch(Preds_ConS, Target_Con)
                        Errors_Pa = torch.sqrt(((S1_hat - Target_Con) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                        Pa_MPs = np.mean(Errors_Pa) * 1000
                        Orig_PaMP0 += Pa_MPs

                        Preds_PelS_T = (Preds_Trans[0]['kp_3d'][center_batch][center_seq][[2], :] + Preds_Trans[0]['kp_3d'][center_batch][center_seq][[3],:]) / 2.0
                        Preds_ConS_T = Preds_Trans[0]['kp_3d'][center_batch][center_seq] - Preds_PelS_T
                        Pose3d_Origin = Preds[0]['kp_3d']-target['kp_3d']
                        Pose3d_Origin = Pose3d_Origin.reshape(Pose3d_Origin.size(0)*Pose3d_Origin.size(1),Pose3d_Origin.size(2),Pose3d_Origin.size(3))
                        Pose3d_Origin = Pose3d_Origin.sum(0)

                        _Preds, Ori_Mat_C2, Ori_Mat_C, Ori_Mat_C2mMP, Ori_Mat_CmMP= Curr_matrix_toTar(self.istrain, 0, 0,Preds_ConS_T,Joint_axis_Filter,retuen_target_3D, Preds_Trans[0]['kp_3d'],count_index_list,origin_count,weight)
                        Pred_PelS = (Ori_Mat_CmMP[[2], :] + Ori_Mat_CmMP[[3], :]) / 2.0
                        Preds_ConS = Ori_Mat_CmMP - Pred_PelS
                        Ori_Dis_Tar = (torch.tensor(Preds_ConS) - Target_Con.cpu()) ** 2
                        Errors_Singe = torch.sqrt(Ori_Dis_Tar.sum(dim=-1)).mean(
                            dim=-1).cpu().numpy()
                        MP_Singe = Errors_Singe * 1000
                        Preds_ConS = torch.tensor(Preds_ConS).cuda().type(torch.float32)
                        Preds_ConS = Preds_ConS.reshape(1, Preds_ConS.size(0), Preds_ConS.size(1))

                        S1_hat = batch_compute_similarity_transform_torch(Preds_ConS, Target_Con)
                        Errors_Pa = torch.sqrt(((S1_hat - Target_Con) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                        Pa_MPs = np.mean(Errors_Pa) * 1000
                        Orig_PaMP1_mMP += Pa_MPs
                        Orig_MP1_mMP += MP_Singe

                        Pred_PelS = (Ori_Mat_C2mMP[[2], :] + Ori_Mat_C2mMP[[3], :]) / 2.0
                        Preds_ConS = Ori_Mat_C2mMP - Pred_PelS
                        Ori_Dis_Tar = (torch.tensor(Preds_ConS) - Target_Con.cpu()) ** 2
                        Errors_Singe = torch.sqrt(Ori_Dis_Tar.sum(dim=-1)).mean(
                            dim=-1).cpu().numpy()
                        MP_Singe = Errors_Singe * 1000
                        Preds_ConS = torch.tensor(Preds_ConS).cuda().type(torch.float32)
                        Preds_ConS = Preds_ConS.reshape(1, Preds_ConS.size(0), Preds_ConS.size(1))
                        S1_hat = batch_compute_similarity_transform_torch(Preds_ConS, Target_Con)
                        Errors_Pa = torch.sqrt(((S1_hat - Target_Con) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                        Pa_MPs = np.mean(Errors_Pa) * 1000
                        Curr_PaMP_mMP += Pa_MPs
                        Curr_MP_mMP += MP_Singe

                        Pred_PelS = (Ori_Mat_C[[2], :] + Ori_Mat_C[[3],:]) / 2.0
                        Preds_ConS = Ori_Mat_C - Pred_PelS
                        Ori_Dis_Tar = (torch.tensor(Preds_ConS) - Target_Con.cpu()) ** 2
                        Errors_Singe = torch.sqrt(Ori_Dis_Tar.sum(dim=-1)).mean(
                            dim=-1).cpu().numpy()
                        MP_Singe = Errors_Singe * 1000
                        Preds_ConS = torch.tensor(Preds_ConS).cuda().type(torch.float32)
                        Preds_ConS = Preds_ConS.reshape(1, Preds_ConS.size(0), Preds_ConS.size(1))
                        S1_hat = batch_compute_similarity_transform_torch(Preds_ConS, Target_Con)
                        Errors_Pa = torch.sqrt(((S1_hat - Target_Con) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                        Pa_MPs = np.mean(Errors_Pa) * 1000
                        Orig_PaMP1 += Pa_MPs
                        Orig_MP1 += MP_Singe

                        Pred_PelS = (Ori_Mat_C2[[2], :] + Ori_Mat_C2[[3], :]) / 2.0
                        Preds_ConS = Ori_Mat_C2 - Pred_PelS
                        apti_dis_targ = ((torch.tensor(Preds_ConS) - Target_Con.cpu()) ** 2)
                        Errors_Singe = torch.sqrt(apti_dis_targ.sum(dim=-1)).mean(
                            dim=-1).cpu().numpy()
                        MP_Singe = Errors_Singe * 1000
                        Preds_ConS = torch.tensor(Preds_ConS).cuda().type(torch.float32)
                        Preds_ConS = Preds_ConS.reshape(1, Preds_ConS.size(0), Preds_ConS.size(1))
                        S1_hat = batch_compute_similarity_transform_torch(Preds_ConS, Target_Con)
                        Errors_Pa = torch.sqrt(((S1_hat - Target_Con) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                        Pa_MPs = np.mean(Errors_Pa) * 1000
                        Curr_PaMP += Pa_MPs
                        Curr_MP += MP_Singe
                        Joint_axis_Opti = Joint_axis_Opti + (Ori_Dis_Tar - apti_dis_targ)

                Orig_MP0_ = Orig_MP0 / Batch_Seq_Num
                Orig_MP1_ = Orig_MP1 / Batch_Seq_Num
                Curr_MP_ = Curr_MP / Batch_Seq_Num

                Orig_MP1_mMP_ = Orig_MP1_mMP / Batch_Seq_Num
                Curr_MP_mMP_ = Curr_MP_mMP / Batch_Seq_Num

                Orig_PaMP0_ = Orig_PaMP0 / Batch_Seq_Num
                Orig_PaMP1_ = Orig_PaMP1 / Batch_Seq_Num
                Curr_PaMP_ = Curr_PaMP / Batch_Seq_Num

                Orig_PaMP1_mMP_ = Orig_PaMP1_mMP / Batch_Seq_Num
                Curr_PaMP_mMP_ = Curr_PaMP_mMP / Batch_Seq_Num

                print('Minimal Original MP:' + str(Orig_MP1_mMP_))
                print('Minimal Currented MP:' + str(Curr_MP_mMP_))

                print('Minimal Original PaMP:' + str(Orig_PaMP1_mMP_))
                print('Minimal Currented PaMP:' + str(Curr_PaMP_mMP_))

                Number_MP+=1
                OrigMP_0_Sum += Orig_MP0_
                OrigMP_1_Sum += Orig_MP1_
                Curr_MP_Sum += Curr_MP_

                OrigMP_1M_Sum += Orig_MP1_mMP_
                Curr_MP_MSum += Curr_MP_mMP_

                Orig_PaMP0_Sum += Orig_PaMP0_
                Orig_PaMP1_Sum += Orig_PaMP1_
                Curr_PaMP_Sum += Curr_PaMP_

                Orig_PaMP1_MSum += Orig_PaMP1_mMP_
                Curr_PaMP_MSum += Curr_PaMP_mMP_


                x = np.arange(len(Pose3d_Origin))

                n_kp = Preds[-1]['kp_3d'].shape[-2]
                pred_j3d = Preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                if(i+128<len(self.test_loader.dataset.vid_indices0)):
                    end_index=i+128
                else:
                    end_index = len(self.test_loader.dataset.vid_indices0)
                for camera_index1 in range(i,end_index):
                    start_end_index=self.test_loader.dataset.vid_indices0[camera_index1]
                    for camera_index2 in range(start_end_index[0],start_end_index[1]+1):
                        camera_list.append(self.test_loader.dataset.camera_list[camera_index2])
                        center_list.append(np.array(self.test_loader.dataset.center_list[camera_index2]))
                        scale_list.append(np.array(self.test_loader.dataset.scale_list[camera_index2]))
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
                self.evaluation_accumulators['camera_para'].append(np.array(camera_list).reshape(len(camera_list), 1))
                self.evaluation_accumulators['center'].append(np.array(center_list))
                self.evaluation_accumulators['scaler'].append(np.array(scale_list))

            # =============>
            batch_time = time.time() - start
            summary_string = f'({i + 1}/{len(self.test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

        print('Finally, The Average Original MP0:' + str(OrigMP_0_Sum/Number_MP))
        print('Finally, The Average Original MP1:' + str(OrigMP_1_Sum/Number_MP))
        print('Finally, The Average Currented MP:' + str(Curr_MP_Sum/Number_MP))
        print('Finally, The Smallest Average Original MP1:' + str(OrigMP_1M_Sum / Number_MP))
        print('Finally, The Smallest Average Currented MP:' + str(Curr_MP_MSum / Number_MP))
        print("Joint and axis errors of all samples")
        print(Joint_axis_Opti)

    def evaluate(self):
        for k, v in self.evaluation_accumulators.items():
            if(k=='pred_j3d' or k=='target_j3d'):
                self.evaluation_accumulators[k] = np.vstack(v)
            else:
                self.evaluation_accumulators[k] = np.concatenate(v,axis=0)

        index_basic=len(self.evaluation_accumulators['pred_j3d'])/8
        index_0,index_1,index_2,index_3,index_4 = int(index_basic)*4,int(index_basic),int(index_basic),int(index_basic),int(index_basic)
        for k, v in self.evaluation_accumulators0.items():
            self.evaluation_accumulators0[k] = []
        for k, v in self.evaluation_accumulators1.items():
            self.evaluation_accumulators1[k] = []
        for k, v in self.evaluation_accumulators2.items():
            self.evaluation_accumulators2[k] = []
        for k, v in self.evaluation_accumulators3.items():
            self.evaluation_accumulators3[k] = []
        for k, v in self.evaluation_accumulators4.items():
            self.evaluation_accumulators4[k] = []
        for k, v in self.evaluation_accumulators_just.items():
            self.evaluation_accumulators_just[k] = []

        for index in range(int(index_basic*8)):
            if(index<index_0):
                self.evaluation_accumulators0['pred_j3d'].append(self.evaluation_accumulators['pred_j3d'][index])
                self.evaluation_accumulators0['target_j3d'].append(self.evaluation_accumulators['target_j3d'][index])
                self.evaluation_accumulators0['camera_para'].append(self.evaluation_accumulators['camera_para'][index])
                self.evaluation_accumulators0['center'].append(self.evaluation_accumulators['center'][index])
                self.evaluation_accumulators0['scaler'].append(self.evaluation_accumulators['scaler'][index])
            elif(index>=index_0 and index<(index_0+index_1)):
                self.evaluation_accumulators1['pred_j3d'].append(self.evaluation_accumulators['pred_j3d'][index])
                self.evaluation_accumulators1['target_j3d'].append(self.evaluation_accumulators['target_j3d'][index])
                self.evaluation_accumulators1['camera_para'].append(self.evaluation_accumulators['camera_para'][index])
                self.evaluation_accumulators1['center'].append(self.evaluation_accumulators['center'][index])
                self.evaluation_accumulators1['scaler'].append(self.evaluation_accumulators['scaler'][index])
            elif(index>=(index_0+index_1) and index<(index_0+index_1+index_2)):
                self.evaluation_accumulators2['pred_j3d'].append(self.evaluation_accumulators['pred_j3d'][index])
                self.evaluation_accumulators2['target_j3d'].append(self.evaluation_accumulators['target_j3d'][index])
                self.evaluation_accumulators2['camera_para'].append(self.evaluation_accumulators['camera_para'][index])
                self.evaluation_accumulators2['center'].append(self.evaluation_accumulators['center'][index])
                self.evaluation_accumulators2['scaler'].append(self.evaluation_accumulators['scaler'][index])
            elif (index >= (index_0+index_1 + index_2) and index < (index_0+index_1 +index_2 + index_3)):
                self.evaluation_accumulators3['pred_j3d'].append(self.evaluation_accumulators['pred_j3d'][index])
                self.evaluation_accumulators3['target_j3d'].append(self.evaluation_accumulators['target_j3d'][index])
                self.evaluation_accumulators3['camera_para'].append(self.evaluation_accumulators['camera_para'][index])
                self.evaluation_accumulators3['center'].append(self.evaluation_accumulators['center'][index])
                self.evaluation_accumulators3['scaler'].append(self.evaluation_accumulators['scaler'][index])
            else:
                self.evaluation_accumulators4['pred_j3d'].append(self.evaluation_accumulators['pred_j3d'][index])
                self.evaluation_accumulators4['target_j3d'].append(self.evaluation_accumulators['target_j3d'][index])
                self.evaluation_accumulators4['camera_para'].append(self.evaluation_accumulators['camera_para'][index])
                self.evaluation_accumulators4['center'].append(self.evaluation_accumulators['center'][index])
                self.evaluation_accumulators4['scaler'].append(self.evaluation_accumulators['scaler'][index])
        count_1,count_2,count_3,count_4=0,0,0,0
        for index in range(int(index_basic)):
            if(self.evaluation_accumulators2['pred_j3d'][index][6][2]-self.evaluation_accumulators2['pred_j3d'][index][2][2]>=0):count_2+=1
            if (self.evaluation_accumulators2['pred_j3d'][index][6][2] -self.evaluation_accumulators2['pred_j3d'][index][3][2] >= 0): count_2 += 1
            if (self.evaluation_accumulators2['pred_j3d'][index][11][2] -self.evaluation_accumulators2['pred_j3d'][index][3][2] >= 0): count_2 += 1
            if (self.evaluation_accumulators2['pred_j3d'][index][11][2] -self.evaluation_accumulators2['pred_j3d'][index][2][2] >= 0): count_2 += 1
            if (self.evaluation_accumulators1['pred_j3d'][index][6][2] -self.evaluation_accumulators1['pred_j3d'][index][2][2] >= 0): count_1 += 1
            if (self.evaluation_accumulators1['pred_j3d'][index][6][2] -self.evaluation_accumulators1['pred_j3d'][index][3][2] >= 0): count_1 += 1
            if (self.evaluation_accumulators1['pred_j3d'][index][11][2] -self.evaluation_accumulators1['pred_j3d'][index][3][2] >= 0): count_1 += 1
            if (self.evaluation_accumulators1['pred_j3d'][index][11][2] -self.evaluation_accumulators1['pred_j3d'][index][2][2] >= 0): count_1 += 1
            max_value_index = [count_1, count_2, count_3, count_4].index(np.array([count_1, count_2]).max())
            if(max_value_index==0):
                self.evaluation_accumulators_just['pred_j3d'].append(self.evaluation_accumulators1['pred_j3d'][index])
                self.evaluation_accumulators_just['target_j3d'].append(self.evaluation_accumulators1['target_j3d'][index])
                self.evaluation_accumulators_just['camera_para'].append(self.evaluation_accumulators1['camera_para'][index])
                self.evaluation_accumulators_just['center'].append(self.evaluation_accumulators1['center'][index])
                self.evaluation_accumulators_just['scaler'].append(self.evaluation_accumulators1['scaler'][index])
            elif(max_value_index==1):
                self.evaluation_accumulators_just['pred_j3d'].append(self.evaluation_accumulators2['pred_j3d'][index])
                self.evaluation_accumulators_just['target_j3d'].append(self.evaluation_accumulators2['target_j3d'][index])
                self.evaluation_accumulators_just['camera_para'].append(self.evaluation_accumulators2['camera_para'][index])
                self.evaluation_accumulators_just['center'].append(self.evaluation_accumulators2['center'][index])
                self.evaluation_accumulators_just['scaler'].append(self.evaluation_accumulators2['scaler'][index])
            elif (max_value_index == 2):
                self.evaluation_accumulators_just['pred_j3d'].append(self.evaluation_accumulators3['pred_j3d'][index])
                self.evaluation_accumulators_just['target_j3d'].append(self.evaluation_accumulators3['target_j3d'][index])
                self.evaluation_accumulators_just['camera_para'].append(self.evaluation_accumulators3['camera_para'][index])
                self.evaluation_accumulators_just['center'].append(self.evaluation_accumulators3['center'][index])
                self.evaluation_accumulators_just['scaler'].append(self.evaluation_accumulators3['scaler'][index])
            else:
                self.evaluation_accumulators_just['pred_j3d'].append(self.evaluation_accumulators4['pred_j3d'][index])
                self.evaluation_accumulators_just['target_j3d'].append(self.evaluation_accumulators4['target_j3d'][index])
                self.evaluation_accumulators_just['camera_para'].append(self.evaluation_accumulators4['camera_para'][index])
                self.evaluation_accumulators_just['center'].append(self.evaluation_accumulators4['center'][index])
                self.evaluation_accumulators_just['scaler'].append(self.evaluation_accumulators4['scaler'][index])

        evaluation_accumulators_target_j3d=[np.array(self.evaluation_accumulators0['target_j3d']),np.array(self.evaluation_accumulators1['target_j3d']),np.array(self.evaluation_accumulators2['target_j3d']),np.array(self.evaluation_accumulators3['target_j3d']),np.array(self.evaluation_accumulators4['target_j3d']),np.array(self.evaluation_accumulators_just['target_j3d'])]
        evaluation_accumulators_target_j3d[2]=evaluation_accumulators_target_j3d[1]
        pred_j3ds = torch.from_numpy(np.array(self.evaluation_accumulators['pred_j3d'])).float()
        target_j3ds = torch.from_numpy(np.array(self.evaluation_accumulators['target_j3d'])).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        Pred_Pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        Tar_Pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0

        pred_j3ds -= Pred_Pelvis
        target_j3ds -= Tar_Pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1))
        errors = errors.mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        Errors_Pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        m2mm = 1000

        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(Errors_Pa) * m2mm
        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            # 'pve': pve,
            'accel': accel,
            'accel_err': accel_err
        }
        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)

    def run(self,cfg,args):
        for index in [0.8]:#[0.8]
            print("Weight:"+str(index))
            for index1 in [24]:#[18,24,30]
                print("The test sample set size:" + str(index1))
                for index2 in [8]:#[2,4,8]
                    print("Number of test samples in each batch:" + str(index2))
                    self.validate(args,index,index1,index2)
        self.evaluate()