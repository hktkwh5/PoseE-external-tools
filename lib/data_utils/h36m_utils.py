import os
import cv2
import glob
import h5py
import json
import joblib
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio
import pickle
from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d
from lib.data_utils.feature_extractor import extract_features
from lib.utils.utils import get_affine_transform
from lib.utils.utils import affine_transform
#以下全从0开始
connectjoint_just=[[9,10],[10,9],[9,8],[8,9],[11,12],[12,11],[12,13],[13,12],[14,15],[15,14],[15,16],[16,15],[1,2],[2,1],[2,3],[3,2],[4,5],[5,4],[5,6],[6,5]]
Inner_circle=[[10,12],[10,15],[12,5],[15,2],[2,5]]
Outer_circle=[[10,13],[10,16],[13,6],[16,3],[3,6]]

def read_openpose(json_file, gt_part, dataset):
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] > 0
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                dist_conf[i] = np.mean(np.sqrt(np.sum(op_conf12*(op_keyp12 - gt_part[:12, :2])**2, axis=1)))
        # closest match
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        if dataset == 'mpii':
            thresh = 30
        elif dataset == 'coco':
            thresh = 10
        else:
            thresh = 0
        # dataset-specific thresholding based on pixel size of person
        if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
            keyp25 = np.zeros([25,3])
        else:
            keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

def read_data_train(dataset_path, debug=False):
    h, w = 256, 256
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    # training data
    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    # product = product(user_list, seq_list, vid_list)
    # user_i, seq_i, vid_i = product[process_id]

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            for j, vid_i in enumerate(vid_list):
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'video_' + str(vid_i))
                # per frame
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                vid_used_frames = []
                vid_used_joints = []
                vid_used_bbox = []
                vid_segments = []
                vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i) + "_vid" + str(vid_i) + "_seg0"
                for i, img_i in tqdm_enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    joints_2d_raw = np.reshape(annot2[vid_i][0][i], (1, 28, 2))
                    joints_2d_raw= np.append(joints_2d_raw, np.ones((1,28,1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "mpii3d",  "spin").reshape((-1,3))

                    # visualize = True
                    # if visualize == True and i == 500:
                    #     import matplotlib.pyplot as plt
                    #
                    #     frame = cv2.cvtColor(cv2.imread(img_i), cv2.COLOR_BGR2RGB)
                    #
                    #     for k in range(49):
                    #         kp = joints_2d[k]
                    #
                    #         frame = cv2.circle(
                    #             frame.copy(),
                    #             (int(kp[0]), int(kp[1])),
                    #             thickness=3,
                    #             color=(255, 0, 0),
                    #             radius=5,
                    #         )
                    #
                    #         cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    #                     (0, 255, 0),
                    #                     thickness=3)
                    #
                    #     plt.imshow(frame)
                    #     plt.show()

                    joints_3d_raw = np.reshape(annot3[vid_i][0][i], (1, 28, 3)) / 1000
                    joints_3d = convert_kps(joints_3d_raw, "mpii3d", "spin").reshape((-1,3))

                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)
                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    # check that all joints are visible
                    x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                    y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < joints_2d.shape[0]:
                        vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                                          str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                        continue

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(img_name.split(".")[0])
                    dataset['img_name'].append(img_i)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_segments.append(vid_uniq_id)
                    vid_used_frames.append(img_i)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

                vid_segments= np.array(vid_segments)
                ids = np.zeros((len(set(vid_segments))+1))
                ids[-1] = len(vid_used_frames) + 1
                if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                    ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1

                for i in tqdm(range(len(set(vid_segments)))):
                    features = extract_features(model, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                                vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                                kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                                dataset='spin', debug=False)
                    dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])
    return dataset

def read_test_data(dataset_path):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
        "valid_i": []
    }

    model = spin.get_pretrained_hmr()

    # user_list = range(1, 7)
    count_iter=0

    # seq_path = os.path.join(dataset_path,
    #                         'mpi_inf_3dhp_test_set',
    #                         'TS' + str(user_i))
    # # mat file with annotations
    # annot_file = os.path.join(seq_path, 'annot_data.mat')
    # mat_as_h5 = h5py.File(annot_file, 'r')
    anno_file = osp.join('H:\\H36M dataset (part)\h36m\\annot\\train_h36m_S9_11.pkl')
    with open(anno_file, 'rb') as f:
        mat_as_h5 = pickle.load(f)



    # annot2 = np.array(mat_as_h5['annot2'])
    # annot3 = np.array(mat_as_h5['univ_annot3'])
    # valid = np.array(mat_as_h5['valid_frame'])
    annot2 = []
    annot3 = []
    valid = []
    frame = []
    #从这里就得筛选了，将合适的样本重新赋值到mat_as_h5数据中，并且重新建立一个mat_as_h5_exp的对象（它用来补充数据集中，它里边的包括一个每项在mat_as_h5中的索引）
    # period_num=int(len(mat_as_h5)/4)
    # mat_as_h5_first=[]
    # mat_as_h5_exp = []
    #以下对每个姿势在4个视图中挑选出可以使用一个或者两个视图
    mat_as_h5_one_hot = []
    count_h36m=[0,0,0,0,0]
    for index1 in range(len(mat_as_h5)):
        #此处循环选择四个视图中最合适的那个
        if(mat_as_h5[index1]['joints_2d'].min()<0 or mat_as_h5[index1]['joints_2d'].min()>1002 or mat_as_h5[index1]['joints_2d'].max()<0 or mat_as_h5[index1]['joints_2d'].max()>1002):
            joint_record=[]
            count_joint=0
            for joint_value in mat_as_h5[index1]['joints_2d']:
                if(joint_value[0]>1002 or joint_value[0]<0 or joint_value[1]<0 or joint_value[1]>1002):
                    joint_record.append(count_joint)
                count_joint += 1
            if(len(joint_record)>=3):
                mat_as_h5_one_hot.append(np.array([0,0,0,0,1]))
                count_h36m[4]+=1
            else:
                if(len(joint_record)==2):
                    if(joint_record in connectjoint_just):
                        mat_as_h5_one_hot.append(np.array([0,0,0,1,0]))
                        count_h36m[3] += 1
                    else:
                        mat_as_h5_one_hot.append(np.array([0,0,1,0,0]))
                        count_h36m[2] += 1
                else:
                    mat_as_h5_one_hot.append(np.array([0,1,0,0,0]))
                    count_h36m[1] += 1
        else:
            mat_as_h5_one_hot.append(np.array([1,0,0,0,0]))
            count_h36m[0] += 1
            continue
    mat_as_h5_one_hot_top = np.array(mat_as_h5_one_hot)

    # mat_as_h5_one_hot_top = np.array(mat_as_h5_one_hot)
    # mat_as_h5_one_hot_sorted = sorted(mat_as_h5_one_hot)
    # x3 = range(len(mat_as_h5_one_hot_sorted))
    # plt.ylim(200, 1200)  # 限定纵轴的范围
    # plt.xlim(0, len(mat_as_h5_one_hot_sorted))
    # plt.plot(x3, mat_as_h5_one_hot_sorted, marker='o', mec='r', mfc='w', label=u'Cora')
    # plt.legend()  # 让图例生效
    # plt.xlabel(u"the number of items", fontsize=15)  # X轴标签
    # plt.ylabel("Inner_Outer_sum", fontsize=13)  # Y轴标签plt.ylabel("Micro/Macro-F;samples;weighted;accuracy")
    # plt.title(u"统计", fontsize=15)  # 标题#plt.title(u"wikipedia(treated)")
    # # plt.savefig('./1.jpg')
    # plt.show()
        # mat_as_h5_con2=[]
        # for index3 in range(len(mat_as_h5_con1)):
        #     continue
        #     # if(mat_as_h5_con1[index3]):
        #
        # #如果没有合适的单视图则使用双视图（尽量为90度夹角的双视图）
        # if(not count):
        #     #这里可以选择视图0和2，或者选择视图7和8. 优先选择这两组中包含人物正面的组
        #     if(period_num[index1] or period_num[index1+period_num]):
        #         if(period_num[index1]):
        #             mat_as_h5_first.append(mat_as_h5[index1])
        #             mat_as_h5_exp.append([mat_as_h5[index1+period_num],index1])
        #         else:
        #             mat_as_h5_first.append(mat_as_h5[index1 + period_num])
        #             mat_as_h5_exp.append([mat_as_h5[index1], index1])
        #     else:
        #         if(mat_as_h5[index1+2*period_num]):
        #             mat_as_h5_first.append(mat_as_h5[index1+2*period_num])
        #             mat_as_h5_exp.append([mat_as_h5[index1+3*period_num],index1])
        #         else:
        #             mat_as_h5_first.append(mat_as_h5[index1 + 3 * period_num])
        #             mat_as_h5_exp.append([mat_as_h5[index1 + 2 * period_num], index1])

    rotation = 0
    Inner_Outer_sum=[]
    mat_as_IO_one_hot=[]
    for index1 in range(len(mat_as_h5)):

        #此处循环选择四个视图中最合适的那个
        data_convert_joint=[]
        for index3 in mat_as_h5[index1]['joints_2d']:
            center=np.array(mat_as_h5[index1]['center']).copy()
            scale=np.array(mat_as_h5[index1]['scale']).copy()
            trans = get_affine_transform(center,float(scale), rotation, np.array([int(256),int(256)]),scale_tmp=300.0)
            data_convert_joint.append(affine_transform(index3[0:2], trans))
        Inner_sum=0
        Outer_sum=0
        for index4 in range(len(Inner_circle)):
            Inner_sum+=abs(data_convert_joint[Inner_circle[index4][0]][0]-data_convert_joint[Inner_circle[index4][1]][0])+abs(data_convert_joint[Inner_circle[index4][0]][1]-data_convert_joint[Inner_circle[index4][1]][1])
            Outer_sum+=abs(data_convert_joint[Outer_circle[index4][0]][0]-data_convert_joint[Outer_circle[index4][1]][0])+abs(data_convert_joint[Outer_circle[index4][0]][1]-data_convert_joint[Outer_circle[index4][1]][1])
        Inner_Outer_sum.append(Inner_sum+Outer_sum)#Inner_Outer_sum.append(Inner_sum+Outer_sum)
    Inner_Outer_sum_sorted=sorted(Inner_Outer_sum)#这里用的升序排序

    x3 = range(len(Inner_Outer_sum))
    plt.ylim(200, 1200)  # 限定纵轴的范围
    plt.xlim(0, len(Inner_Outer_sum))
    plt.plot(x3, Inner_Outer_sum_sorted, marker='o', mec='r', mfc='w', label=u'Cora')
    plt.legend()  # 让图例生效
    plt.xlabel(u"the number of items", fontsize=15)  # X轴标签
    plt.ylabel("Inner_Outer_sum", fontsize=13)  # Y轴标签plt.ylabel("Micro/Macro-F;samples;weighted;accuracy")
    plt.title(u"统计", fontsize=15)  # 标题#plt.title(u"wikipedia(treated)")
    # plt.savefig('./1.jpg')
    plt.show()

    Inner_Outer_sum_sorted=[[Inner_Outer_sum_sorted[0:int(len(Inner_Outer_sum_sorted)/12)]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12):int(len(Inner_Outer_sum_sorted)/12)*2]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*2:int(len(Inner_Outer_sum_sorted)/12)*3]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*3:int(len(Inner_Outer_sum_sorted)/12)*4]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*4:int(len(Inner_Outer_sum_sorted)/12)*5]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*5:int(len(Inner_Outer_sum_sorted)/12)*6]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*6:int(len(Inner_Outer_sum_sorted)/12)*7]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*7:int(len(Inner_Outer_sum_sorted)/12)*8]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*8:int(len(Inner_Outer_sum_sorted)/12)*9]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*9:int(len(Inner_Outer_sum_sorted)/12)*10]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*10:int(len(Inner_Outer_sum_sorted)/12)*11]],[Inner_Outer_sum_sorted[int(len(Inner_Outer_sum_sorted)/12)*11:]]]
    for index1 in range(len(mat_as_h5)):
        count_mask=0
        for index2 in Inner_Outer_sum_sorted:
            count_mask+=1
            if(Inner_Outer_sum[index1]>=index2[0][0] and Inner_Outer_sum[index1]<=index2[0][int(len(index2[0]))-1]):
                break
        if(count_mask==1):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
        elif(count_mask==2):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
        elif(count_mask==3):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
        elif(count_mask==4):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
        elif (count_mask == 5):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
        elif (count_mask == 6):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
        elif (count_mask == 7):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        elif (count_mask == 8):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        elif (count_mask == 9):
            mat_as_IO_one_hot.append(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        elif (count_mask == 10):
            mat_as_IO_one_hot.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        elif (count_mask == 11):
            mat_as_IO_one_hot.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        elif (count_mask == 12):
            mat_as_IO_one_hot.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        else:
            print("程序有错误!")
            exit()
    if(count_iter==0):
        mat_as_IO_one_hot_top = np.array(mat_as_IO_one_hot)
    else:
        mat_as_IO_one_hot_top = np.concatenate((mat_as_IO_one_hot_top, np.array(mat_as_IO_one_hot)),axis=0)
    count_iter += 1

    

    for index_value in mat_as_h5:
        annot2.append(index_value['joints_2d'])#这是像素坐标，原点是图像左上角
        annot3.append(index_value['joints_3d'])#这是相机坐标，原点是图像中心
        valid.append(index_value['image'].split('\\')[0])
        frame.append(index_value['image'].split('\\')[1])
    annot2 = np.array(annot2)
    annot3 = np.array(annot3)
    valid = np.array(valid)
    frame = np.array(frame)
    vid_used_frames = []
    vid_used_joints = []
    vid_used_bbox = []
    # vid_segments = []
    # vid_uniq_id = "subj" + str(user_i) + "_seg0"
    for num in tqdm(enumerate(valid)):
        frame_i, valid_i=frame[num[0]],valid[num[0]]
        # img_i = os.path.join('mpi_inf_3dhp_test_set',
        #                         'TS' + str(user_i),
        #                         'imageSequence',
        #                         'img_' + str(frame_i + 1).zfill(6) + '.jpg')
        img_i = os.path.join('C:\\H36M\\',
                             valid_i+'\\',
                             frame_i)

        joints_2d_raw = np.expand_dims(annot2[num[0], :, :], axis = 0)#joints_2d_raw = np.expand_dims(annot2[num[0], 0, :, :], axis = 0)，，这里边全都是像素坐标一共就17个关节点
        joints_2d_raw = np.append(joints_2d_raw, np.ones((1, 17, 1)), axis=2)#这里给joints_2d_raw增加了一个z维，但是都赋值为1
        joints_2d = convert_kps(joints_2d_raw, src="h36m", dst="spin").reshape((-1, 3))#最后得到了关节顺序与spin一样的关节顺序，但是中间可能有间隔，也不是从第一个开始

        # visualize = True
        # if visualize == True:
        #     import matplotlib.pyplot as plt
        #
        #     frame = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, img_i)), cv2.COLOR_BGR2RGB)
        #
        #     for k in range(49):
        #         kp = joints_2d[k]
        #
        #         frame = cv2.circle(
        #             frame.copy(),
        #             (int(kp[0]), int(kp[1])),
        #             thickness=3,
        #             color=(255, 0, 0),
        #             radius=5,
        #         )
        #
        #         cv2.putText(frame, f'{k}', (int(kp[0]), int(kp[1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0),
        #                     thickness=3)
        #
        #     plt.imshow(frame)
        #     plt.show()

        joints_3d_raw = np.reshape(annot3[num[0], :, :], (1, 17, 3)) / 1000#joints_3d_raw = np.reshape(annot3[num[0], 0, :, :], (1, 17, 3)) / 1000
        joints_3d = convert_kps(joints_3d_raw, "h36m", "spin").reshape((-1, 3))#最后得到了关节顺序与spin一样的关节顺序，但是中间可能有间隔，也不是从第一个开始
        joints_3d = joints_3d - joints_3d[39] # substract pelvis zero is the root for test，这里是所有的元素都减了joints_3d[39]的这个浮点数，第39个就是spin中的root

        bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)#这里的box用的像素坐标，

        # check that all joints are visible
        # img_file = os.path.join(dataset_path, img_i)
        img_file = img_i
        I = cv2.imread(img_file)
        h, w, _ = I.shape
        # x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
        # y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
        # ok_pts = np.logical_and(x_in, y_in)

        # if np.sum(ok_pts) < joints_2d.shape[0]:
        #     vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1]) + "_seg" + \
        #                   str(int(dataset['vid_name'][-1].split("_")[-1][3:]) + 1)
        #     continue


        # dataset['vid_name'].append(vid_uniq_id)
        # dataset['frame_id'].append(img_file.split("/")[-1].split(".")[0])
        dataset['frame_id'].append(img_file.split("\\")[-1].split(".")[0])
        dataset['img_name'].append(img_file)
        dataset['joints2D'].append(joints_2d)
        dataset['joints3D'].append(joints_3d)
        dataset['bbox'].append(bbox)
        dataset['valid_i'].append(valid_i)

        # vid_segments.append(vid_uniq_id)
        vid_used_frames.append(img_file)
        vid_used_joints.append(joints_2d)
        vid_used_bbox.append(bbox)

    # vid_segments = np.array(vid_segments)

    # ids[-1] = len(vid_used_frames) + 1
    # if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
    #     ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1
    index_1000=len(set(vid_used_frames))
    for i in tqdm(range(0,index_1000,1000)):
        if(index_1000-i<=1000):
            features = extract_features(model, np.array(vid_used_frames)[i:index_1000],
                                        vid_used_bbox[i:index_1000],
                                        kp_2d=np.array(vid_used_joints)[i:index_1000],
                                        dataset='spin', debug=False)
            dataset['features'].append(features)  # 这里所得到的特征是正确的
            break
        else:
            features = extract_features(model, np.array(vid_used_frames)[i:i+1000],
                                        vid_used_bbox[i:i+1000],
                                        kp_2d=np.array(vid_used_joints)[i:i+1000],
                                        dataset='spin', debug=False)
            dataset['features'].append(features)#这里所得到的特征是正确的

    save_file1 = pd.DataFrame(data=mat_as_h5_one_hot_top)
    save_file2 = pd.DataFrame(data=mat_as_IO_one_hot_top)
    # save_file1.to_csv('C:\\H36M\\video_dir\\iris_training1_H36M_S9_11.csv')
    # save_file2.to_csv('C:\\H36M\\video_dir\\iris_training2_H36M_S9_11.csv')
    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='C:\\MPI-INF-3DHP(beifen)\\MPI-INF-3DHP(beifen)')
    args = parser.parse_args()
    dataset = read_test_data(args.dir)
    # joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'mpii3d_val_db_H36M_S9_11_bgr.pt'))

    # dataset = read_data_train(args.dir)
    # joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'mpii3d_train_db.pt'))



