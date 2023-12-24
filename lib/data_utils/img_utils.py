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
import cv2
import torch

import random
import numpy as np
import torchvision.transforms as transforms
from skimage.util.shape import view_as_windows

def get_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def do_augmentation(scale_factor=0.3, color_factor=0.2):
    scale = random.uniform(1.1, 1.1+scale_factor)#scale = random.uniform(1.2, 1.2+scale_factor)
    # scale = np.clip(np.random.randn(), 0.0, 1.0) * scale_factor + 1.2
    rot = 0 # np.clip(np.random.randn(), -2.0, 2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    do_flip = False # aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return img_patch, trans

def crop_image(image, kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    # get augmentation params
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.1, 0, False, [1.0, 1.0, 1.0]#scale, rot, do_flip, color_scale = 1.3, 0, False, [1.0, 1.0, 1.0]

    # generate image patch
    image, trans = generate_patch_image_cv(
        image,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)
    return image, kp_2d, trans

def transfrom_keypoints(kp_2d, center_x, center_y, width, height, patch_width, patch_height, do_augment):

    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation()
    else:
        scale, rot, do_flip, color_scale = 1.1,0, False, [1.0, 1.0, 1.0]#scale, rot, do_flip, color_scale = 1.2, 0, False, [1.0, 1.0, 1.0]

    # generate transformation
    trans = gen_trans_from_patch_cv(
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        scale,
        rot,
        inv=False,
    )

    for n_jt in range(kp_2d.shape[0]):
        kp_2d[n_jt] = trans_point2d(kp_2d[n_jt], trans)

    return kp_2d, trans

def get_image_crops(image_file, bboxes):
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    crop_images = []
    for bb in bboxes:
        c_y, c_x = (bb[0]+bb[2]) // 2, (bb[1]+bb[3]) // 2
        h, w = bb[2]-bb[0], bb[3]-bb[1]
        w = h = np.where(w / h > 1, w, h)
        crop_image, _ = generate_patch_image_cv(
            cvimg=image.copy(),
            c_x=c_x,
            c_y=c_y,
            bb_width=w,
            bb_height=h,
            patch_width=224,
            patch_height=224,
            do_flip=False,
            scale=1.1,#scale=1.3,
            rot=0,
        )
        crop_image = convert_cvimg_to_tensor(crop_image)
        crop_images.append(crop_image)

    batch_image = torch.cat([x.unsqueeze(0) for x in crop_images])
    return batch_image

def get_single_image_crop(image, bbox, scale=1.1):#def get_single_image_crop(image, bbox, scale=1.2):
    if isinstance(image, str):
        image = "I:\\diskF" + image[2:]
        if os.path.isfile(image):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, _ = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=224,
        patch_height=224,
        do_flip=False,
        scale=scale,
        rot=0,
    )

    crop_image = convert_cvimg_to_tensor(crop_image)

    return crop_image

def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.1, crop_size=224):#def get_single_image_crop_demo(image, bbox, kp_2d, scale=1.2, crop_size=224):
    if isinstance(image, str):
        if os.path.isfile(image):
            image_= cv2.imread(image)
            image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        else:
            print(image)
            raise BaseException(image, 'is not a valid file!')
    elif isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise('Unknown type for object', type(image))

    crop_image, trans = generate_patch_image_cv(
        cvimg=image.copy(),
        c_x=bbox[0],
        c_y=bbox[1],
        bb_width=bbox[2],
        bb_height=bbox[3],
        patch_width=crop_size,
        patch_height=crop_size,
        do_flip=False,
        scale=scale,
        rot=0,
    )
    if kp_2d is not None:
        for n_jt in range(kp_2d.shape[0]):
            kp_2d[n_jt, :2] = trans_point2d(kp_2d[n_jt], trans)
    raw_image = crop_image.copy()
    crop_image = convert_cvimg_to_tensor(crop_image)
    return crop_image, raw_image, kp_2d

def read_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    return convert_cvimg_to_tensor(image)

def convert_cvimg_to_tensor(image):
    transform = get_default_transform()
    image = transform(image)
    return image

def torch2numpy(image):
    image = image.detach().cpu()
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    image = inv_normalize(image)
    image = image.clamp(0., 1.)
    image = image.numpy() * 255.
    image = np.transpose(image, (1, 2, 0))
    return image.astype(np.uint8)

def torch_vid2numpy(video):
    video = video.detach().cpu().numpy()
    # video = np.transpose(video, (0, 2, 1, 3, 4)) # NCTHW->NTCHW
    # Denormalize
    mean = np.array([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255])
    std = np.array([1 / 0.229, 1 / 0.224, 1 / 0.255])
    mean = mean[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    std = std[np.newaxis, np.newaxis, ..., np.newaxis, np.newaxis]
    video = (video - mean) / std # [:, :, i, :, :].sub_(mean[i]).div_(std[i]).clamp_(0., 1.).mul_(255.)
    video = video.clip(0.,1.) * 255
    video = video.astype(np.uint8)
    return video

def get_bbox_from_kp2d(kp_2d):
    # get bbox
    if len(kp_2d.shape) > 2:
        ul = np.array([kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)])  # upper left
        lr = np.array([kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)])  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1
    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox

def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d

def get_default_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return transform

def split_into_chunks(vid_names, seqlen, stride):
    video_start_end_indices1 = []
    video_names, group = np.unique(vid_names, return_index=True)#这个地方只要保持4个4个的一组就能将视图给分出来，在mpi的数据集中
    group_con = group
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]
    indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])

    for idx in range(len(video_names)):
        indexes = indices[idx]
        if indexes.shape[0] < seqlen:
            continue
        chunks = view_as_windows(indexes, (seqlen,), step=stride)
        start_finish = chunks[:, (0, -1)].tolist()
        video_start_end_indices1 += start_finish
    print(video_start_end_indices1[-1])

    four_view_index = []
    for index in range(0, int(len(video_names) / 4)):
        four_view_index_con = []
        for index1 in range(4):
            if (index * 4 + index1 == int(len(video_names)) - 1):
                index_end = group_con[index * 4 + index1] + four_view_index_con[0][1] - four_view_index_con[0][0]
                four_view_index_con.append([group_con[index * 4 + index1], index_end])
            else:
                index_end = np.where(group == group_con[index * 4 + index1])[0][0]
                four_view_index_con.append([group_con[index * 4 + index1], group[index_end + 1]])
        four_view_index.append(four_view_index_con)

    # indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])#将5500按不同视角分成4组，indices中是被分成四个array的实际内容
    video_start_end_indices2 = []
    for index in range(len(four_view_index)):
        video_start_end_indices2.append([])
        for idx in range(len(four_view_index[0])):  # 循环4个视图
            video_start_end_indices2[index].append([])
            # indexes = four_view_index[index][idx][1]-four_view_index[index][idx][0]#取一个视图的内容
            for index1 in range(four_view_index[index][idx][0], four_view_index[index][idx][1], seqlen):
                if (index1 + seqlen - 1 < four_view_index[index][idx][1]):
                    video_start_end_indices2[index][idx].append([index1, index1 + seqlen - 1])
            # if indexes.shape[0] < seqlen:#如果一个都没有就直接跳到下一个视角
            #     continue
            # chunks = view_as_windows(indexes, (seqlen,), step=stride)#input_array : 要切割的数据；window_shape:切割窗口大小；step：窗口移动的步幅
            # start_finish = chunks[:, (0, -1)].tolist()
            # video_start_end_indices += start_finish
    print(video_start_end_indices2[-1])  # 输出这个索引的最后一项
    return video_start_end_indices1,video_start_end_indices2

def MPI_best_double_view_chunks(vid_names, seqlen, stride,iris_training1,iris_training2,iris_training3):#(5500),16,16
    # video_start_end_indices = []
    video_names, group = np.unique(vid_names, return_index=True)#np.unique函数是去掉vid_names中的重复项，并将它们组合成四个组，也就是四个视觉，video_names放唯一的四个项，group放这四个项的其实索引（比如（0，1375，2750，4125））
    group_con=group
    # view_counts=len(video_names)
    perm = np.argsort(group)#排序并返回排序的索引
    video_names, group = video_names[perm], group[perm]

    four_view_index = []
    for index in range(0, int(len(video_names) / 4) ):
        four_view_index_con = []
        for index1 in range(4):
            if(index * 4 + index1==int(len(video_names))-1):
                index_end = group_con[index * 4 + index1]+four_view_index_con[0][1]-four_view_index_con[0][0]
                four_view_index_con.append([group_con[index * 4 + index1], index_end])
            else:
                index_end=np.where(group == group_con[index * 4 + index1])[0][0]
                four_view_index_con.append([group_con[index * 4 + index1], group[index_end+1]])
        four_view_index.append(four_view_index_con)

    # indices = np.split(np.arange(0, vid_names.shape[0]), group[1:])#将5500按不同视角分成4组，indices中是被分成四个array的实际内容
    video_start_end_indices=[]
    for index in range(len(four_view_index)):
        video_start_end_indices.append([])
        for idx in range(len(four_view_index[0])):#循环4个视图
            video_start_end_indices[index].append([])
            # indexes = four_view_index[index][idx][1]-four_view_index[index][idx][0]#取一个视图的内容
            for index1 in range(four_view_index[index][idx][0],four_view_index[index][idx][1],seqlen):
                if(index1+seqlen-1<four_view_index[index][idx][1]):
                    video_start_end_indices[index][idx].append([index1,index1+seqlen-1])
            # if indexes.shape[0] < seqlen:#如果一个都没有就直接跳到下一个视角
            #     continue
            # chunks = view_as_windows(indexes, (seqlen,), step=stride)#input_array : 要切割的数据；window_shape:切割窗口大小；step：窗口移动的步幅
            # start_finish = chunks[:, (0, -1)].tolist()
            # video_start_end_indices += start_finish
    print(video_start_end_indices[-1])#输出这个索引的最后一项

    #以下对video_start_end_indices中的所有项进行筛选，因为video_start_end_indices中存储的是四个视图的相同内容，我们需要从中挑选出至少两个视图的起始索引
    len_singe_view_counts=int(len(four_view_index))
    just_index_mulview1 = []
    just_index_mulview2 = []
    for index in range(len(video_start_end_indices)):
        just_index_mulview1.append([]),just_index_mulview2.append([])
        for index1 in range(len(video_start_end_indices[0])):
            just_index_mulview1[index].append([]), just_index_mulview2[index].append([])
            for index2 in range(len(video_start_end_indices[index][index1])):
                just_index_mulview1[index][index1].append(True)
                just_index_mulview2[index][index1].append(0)

    for index in range(len(video_start_end_indices)):#动作
        for index1 in range(len(video_start_end_indices[0])):#视图
            for index2 in range(len(video_start_end_indices[index][index1])):
                for index3 in range(video_start_end_indices[index][index1][index2][0],video_start_end_indices[index][index1][index2][1]+1):
                    if (iris_training1[index3] == [0, 0, 0, 1, 0] or iris_training1[index3] == [0, 0, 0, 0, 1]):
                        just_index_mulview1[index][index1][index2]=False

                    if (iris_training2[index3] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] or iris_training2[index3] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                        just_index_mulview2[index][index1][index2] += 6
                    elif (iris_training2[index3] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] or iris_training2[index3] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]):
                        just_index_mulview2[index][index1][index2] += 5
                    elif (iris_training2[index3] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] or iris_training2[index3] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]):
                        just_index_mulview2[index][index1][index2] += 4
                    elif (iris_training2[index3] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] or iris_training2[index3] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
                        just_index_mulview2[index][index1][index2] += 3
                    elif (iris_training2[index3] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] or iris_training2[index3] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]):
                        just_index_mulview2[index][index1][index2] += 2
                    else:
                        just_index_mulview2[index][index1][index2] += 1
    First_priority_single_view = []
    Second_priority_single_view = []
    #以下根据可以决定视角的两个列表对四个视觉进行筛选，将筛选的最好的视图索引放到First_priority_single_view列表中，而将其次的视图放到Second_priority_single_view列表中。
    for index in range(len(video_start_end_indices)):
        # First_priority_single_view.append([]),Second_priority_single_view.append([])
        for index1 in range(len(video_start_end_indices[index][0])):
            count = 0
            for index2 in range(len(video_start_end_indices[index])):
                if(just_index_mulview1[index][index2][index1]):
                    count += 1
                    # First_priority_single_view.append(video_start_end_indices[index][index2][index1])
                    # Second_priority_single_view.append(video_start_end_indices[index][index2][index1])
            if(count>=2):
                max_value, max_index = 0, 0
                for index2 in range(len(video_start_end_indices[index])):
                    if (just_index_mulview2[index][index2][index1] > max_value and just_index_mulview1[index][index2][index1]):
                        max_value = just_index_mulview2[index][index2][index1]
                        max_index = index2
                First_priority_single_view.append(video_start_end_indices[index][max_index][index1])
                just_index_mulview2[index][max_index][index1]=0
                max_value, max_index = 0, 0
                for index2 in range(len(video_start_end_indices[index])):
                    if (just_index_mulview2[index][index2][index1] > max_value and just_index_mulview1[index][index2][index1]):
                        max_value = just_index_mulview2[index][index2][index1]
                        max_index = index2
                Second_priority_single_view.append(video_start_end_indices[index][max_index][index1])

            elif(count==1):
                max_value, max_index = 0, 0
                for index2 in range(len(video_start_end_indices[index])):
                    if(just_index_mulview1[index][index2][index1]):
                        First_priority_single_view.append(video_start_end_indices[index][index2][index1])
                    if(not just_index_mulview1[index][index2][index1] and just_index_mulview2[index][index2][index1] > max_value):
                        max_value = just_index_mulview2[index][index2][index1]
                        max_index = index2
                Second_priority_single_view.append(video_start_end_indices[index][max_index][index1])
            else:
                for index2 in range(len(video_start_end_indices[index])):
                    if (just_index_mulview2[index][index2][index1] > max_value):
                        max_value = just_index_mulview2[index][index2][index1]
                        max_index = index2
                First_priority_single_view.append(video_start_end_indices[index][max_index][index1])
                just_index_mulview2[index][max_index][index1]=0
                max_value, max_index = 0, 0
                for index2 in range(len(video_start_end_indices[index])):
                    if (just_index_mulview2[index][index2][index1] > max_value):
                        max_value = just_index_mulview2[index][index2][index1]
                        max_index = index2
                Second_priority_single_view.append(video_start_end_indices[index][max_index][index1])
    # return video_start_end_indices
    return First_priority_single_view,Second_priority_single_view
def MPI_best_double_view_chunks_mode3(dataset, seqlen, stride,iris_training3):
    vid_names = []
    for index in range(len(dataset)):
        # dataset['valid_i'][index] = dataset[index].split('_')[5]
        vid_names.append(dataset[index].split('_')[5])
    vid_names = np.array(vid_names)

    video_names, group = np.unique(vid_names,
                                   return_index=True)  # np.unique函数是去掉vid_names中的重复项，并将它们组合成四个组，也就是四个视觉，video_names放唯一的四个项，group放这四个项的其实索引（比如（0，1375，2750，4125））
    # view_counts=len(video_names)
    four_view_index = [[], [], [], []]
    for index1 in range(len(dataset)):
        index1_con = list(video_names).index(dataset[index1].split('_')[5])
        four_view_index[index1_con].append(index1)
        # index2_con=dataset['img_name'][index2].split('.')[0]+dataset['img_name'][index2].split('.')[1].split('_')[1]
        # if(count==4):four_view_index[0].append(con_index[0]),four_view_index[1].append(con_index[1]),four_view_index[2].append(con_index[2]),four_view_index[3].append(con_index[3])
    four_view_index_seqlen = [[], [], [], []]
    for index in range(0, len(four_view_index[0]), seqlen):
        if (((index + seqlen - 1) < len(four_view_index[0]) and four_view_index[0][index + seqlen - 1] -
             four_view_index[0][index]) == (seqlen - 1) and (
                four_view_index[1][index + seqlen - 1] - four_view_index[1][index]) == (seqlen - 1) and (
                four_view_index[2][index + seqlen - 1] - four_view_index[2][index]) == (seqlen - 1) and (
                four_view_index[3][index + seqlen - 1] - four_view_index[3][index]) == (seqlen - 1)):
            four_view_index_seqlen[0].append([four_view_index[0][index], four_view_index[0][index + seqlen - 1]])
            four_view_index_seqlen[1].append([four_view_index[1][index], four_view_index[1][index + seqlen - 1]])
            four_view_index_seqlen[2].append([four_view_index[2][index], four_view_index[2][index + seqlen - 1]])
            four_view_index_seqlen[3].append([four_view_index[3][index], four_view_index[3][index + seqlen - 1]])
    just_index_mulview2 = []
    for index in range(len(four_view_index_seqlen)):
        just_index_mulview2.append([])
        for index1 in range(len(four_view_index_seqlen[0])):
            just_index_mulview2[index].append(0)

    for index in range(len(four_view_index_seqlen)):
        for index1 in range(len(four_view_index_seqlen[0])):
            for index3 in range(four_view_index_seqlen[index][index1][0], four_view_index_seqlen[index][index1][1] + 1):
                if (iris_training3[index3] == [1, 0, 0, 0]):
                    just_index_mulview2[index][index1] += 7
                elif (iris_training3[index3] == [0, 1, 0, 0]):
                    just_index_mulview2[index][index1] += 5
                elif (iris_training3[index3] == [0, 0, 1, 0]):
                    just_index_mulview2[index][index1] += 3
                else:
                    just_index_mulview2[index][index1] += 1
    First_priority_single_view = []
    Second_priority_single_view = []
    Third_priority_single_view = []
    fourth_priority_single_view = []
    max_index, max_value = 0, 0
    # 以下根据可以决定视角的两个列表对四个视觉进行筛选，将筛选的最好的视图索引放到First_priority_single_view列表中，而将其次的视图放到Second_priority_single_view列表中。
    for index in range(len(four_view_index_seqlen[index])):
        for index1 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index1][index] > max_value):
                max_value = just_index_mulview2[index1][index]
                max_index = index1
        First_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index] = 0
        max_value, max_index = 0, 0
        for index2 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index2][index] > max_value):
                max_value = just_index_mulview2[index2][index]
                max_index = index2
        Second_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index] = 0
        max_value, max_index = 0, 0
        for index3 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index3][index] > max_value):
                max_value = just_index_mulview2[index3][index]
                max_index = index3
        Third_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index] = 0
        max_value, max_index = 0, 0
        for index4 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index4][index] > max_value):
                max_value = just_index_mulview2[index4][index]
                max_index = index4
        fourth_priority_single_view.append(four_view_index_seqlen[max_index][index])

    return_sum_index = four_view_index_seqlen[0] + four_view_index_seqlen[1] + four_view_index_seqlen[2] + \
                       four_view_index_seqlen[3]
    # return video_start_end_indices
    return return_sum_index, First_priority_single_view, Second_priority_single_view, Third_priority_single_view, fourth_priority_single_view

def H36M_best_double_view_chunks(dataset, seqlen, stride,iris_training1,iris_training2,iris_training3):#(5500),16,16
    vid_names=[]
    for index in range(len(dataset['img_name'])):
        dataset['valid_i'][index] = dataset['img_name'][index].split('.')[1].split('_')[0]
        vid_names.append(dataset['img_name'][index].split('.')[1].split('_')[0])
    vid_names=np.array(vid_names)

    video_names, group = np.unique(vid_names, return_index=True)#np.unique函数是去掉vid_names中的重复项，并将它们组合成四个组，也就是四个视觉，video_names放唯一的四个项，group放这四个项的其实索引（比如（0，1375，2750，4125））
    # view_counts=len(video_names)
    four_view_index = [[],[],[],[]]
    for index1 in range(len(dataset['img_name'])):
        index1_con = list(video_names).index(dataset['img_name'][index1].split('.')[1].split('_')[0])
        four_view_index[index1_con].append(index1)
        # index2_con=dataset['img_name'][index2].split('.')[0]+dataset['img_name'][index2].split('.')[1].split('_')[1]
        # if(count==4):four_view_index[0].append(con_index[0]),four_view_index[1].append(con_index[1]),four_view_index[2].append(con_index[2]),four_view_index[3].append(con_index[3])
    four_view_index_seqlen = [[], [], [], []]
    for index in range(0,len(four_view_index[0]),seqlen):
        if(((index+seqlen-1)<len(four_view_index[0]) and four_view_index[0][index+seqlen-1]-four_view_index[0][index])==(seqlen-1) and (four_view_index[1][index+seqlen-1]-four_view_index[1][index])==(seqlen-1) and (four_view_index[2][index+seqlen-1]-four_view_index[2][index])==(seqlen-1) and (four_view_index[3][index+seqlen-1]-four_view_index[3][index])==(seqlen-1)):
            four_view_index_seqlen[0].append([four_view_index[0][index],four_view_index[0][index+seqlen-1]])
            four_view_index_seqlen[1].append([four_view_index[1][index], four_view_index[1][index + seqlen - 1]])
            four_view_index_seqlen[2].append([four_view_index[2][index], four_view_index[2][index + seqlen - 1]])
            four_view_index_seqlen[3].append([four_view_index[3][index], four_view_index[3][index + seqlen - 1]])
    just_index_mulview2 = []
    for index in range(len(four_view_index_seqlen)):
        just_index_mulview2.append([])
        for index1 in range(len(four_view_index_seqlen[0])):
            just_index_mulview2[index].append(0)

    for index in range(len(four_view_index_seqlen)):
        for index1 in range(len(four_view_index_seqlen[0])):
            for index3 in range(four_view_index_seqlen[index][index1][0],four_view_index_seqlen[index][index1][1]+1):
                if (iris_training3[index3] == [1, 0, 0, 0]):
                    just_index_mulview2[index][index1] += 7
                elif (iris_training3[index3] == [0, 1, 0, 0]):
                    just_index_mulview2[index][index1] += 5
                elif (iris_training3[index3] == [0, 0, 1, 0]):
                    just_index_mulview2[index][index1] += 3
                else:
                    just_index_mulview2[index][index1] += 1
    First_priority_single_view = []
    Second_priority_single_view = []
    Third_priority_single_view = []
    fourth_priority_single_view = []
    max_index,max_value=0,0
    #以下根据可以决定视角的两个列表对四个视觉进行筛选，将筛选的最好的视图索引放到First_priority_single_view列表中，而将其次的视图放到Second_priority_single_view列表中。
    for index in range(len(four_view_index_seqlen[index])):
        for index1 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index1][index] > max_value):
                max_value = just_index_mulview2[index1][index]
                max_index = index1
        First_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index]=0
        max_value, max_index = 0, 0
        for index2 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index2][index] > max_value):
                max_value = just_index_mulview2[index2][index]
                max_index = index2
        Second_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index] = 0
        max_value, max_index = 0, 0
        for index3 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index3][index] > max_value):
                max_value = just_index_mulview2[index3][index]
                max_index = index3
        Third_priority_single_view.append(four_view_index_seqlen[max_index][index])
        just_index_mulview2[max_index][index] = 0
        max_value, max_index = 0, 0
        for index4 in range(len(four_view_index_seqlen)):
            if (just_index_mulview2[index4][index] > max_value):
                max_value = just_index_mulview2[index4][index]
                max_index = index4
        fourth_priority_single_view.append(four_view_index_seqlen[max_index][index])

    return_sum_index=four_view_index_seqlen[0]+four_view_index_seqlen[1]+four_view_index_seqlen[2]+four_view_index_seqlen[3]
    # return video_start_end_indices
    return return_sum_index,First_priority_single_view,Second_priority_single_view,Third_priority_single_view,fourth_priority_single_view