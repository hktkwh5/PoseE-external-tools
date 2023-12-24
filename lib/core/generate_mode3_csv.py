import torch
import pandas as pd
import numpy as np
def generate_mode3_csv_method(evaluation_accumulators0,index_basic,test_loader):
    item_counts=len(test_loader.dataset.db['features'])
    vid_indices_con=[]
    vid_indices=test_loader.dataset.vid_indices0
    for index in vid_indices:
        for index1 in range(index[0],index[1]+1):
            vid_indices_con.append(index1)
    intervel_count=int(len(vid_indices_con)/4)

    mode3_camera_index=[[0, 0, 0, 0]]*item_counts
    index_basic=int(index_basic)
    evaluation_accumulators_pred_j3d=[np.array(evaluation_accumulators0['pred_j3d'][0:index_basic]),np.array(evaluation_accumulators0['pred_j3d'][index_basic:2*index_basic]),np.array(evaluation_accumulators0['pred_j3d'][2*index_basic:3*index_basic]),np.array(evaluation_accumulators0['pred_j3d'][3*index_basic:])]
    evaluation_accumulators_target_j3d=[np.array(evaluation_accumulators0['target_j3d'][0:index_basic]),np.array(evaluation_accumulators0['target_j3d'][index_basic:2*index_basic]),np.array(evaluation_accumulators0['target_j3d'][2*index_basic:3*index_basic]),np.array(evaluation_accumulators0['target_j3d'][3*index_basic:])]
    camera_count=int(len(evaluation_accumulators0['pred_j3d'])/index_basic)
    # mode3_camera_index1,mode3_camera_index2,mode3_camera_index3,mode3_camera_index4=[],[],[],[]
    # errors=[[],[],[],[]]
    for index1 in range(index_basic):
        pred_j3ds1 = torch.from_numpy(evaluation_accumulators_pred_j3d[0][index1]).float()
        target_j3ds1 = torch.from_numpy(evaluation_accumulators_target_j3d[0][index1]).float()
        pred_pelvis = (pred_j3ds1[ [2], :] + pred_j3ds1[ [3], :]) / 2.0
        target_pelvis = (target_j3ds1[ [2], :] + target_j3ds1[ [3], :]) / 2.0
        pred_j3ds1 -= pred_pelvis
        target_j3ds1 -= target_pelvis
        errors1=torch.sqrt(((pred_j3ds1 - target_j3ds1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        pred_j3ds1 = torch.from_numpy(evaluation_accumulators_pred_j3d[1][index1]).float()
        target_j3ds1 = torch.from_numpy(evaluation_accumulators_target_j3d[1][index1]).float()
        pred_pelvis = (pred_j3ds1[ [2], :] + pred_j3ds1[ [3], :]) / 2.0
        target_pelvis = (target_j3ds1[ [2], :] + target_j3ds1[ [3], :]) / 2.0
        pred_j3ds1 -= pred_pelvis
        target_j3ds1 -= target_pelvis
        errors2 = torch.sqrt(((pred_j3ds1 - target_j3ds1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        pred_j3ds1 = torch.from_numpy(evaluation_accumulators_pred_j3d[2][index1]).float()
        target_j3ds1 = torch.from_numpy(evaluation_accumulators_target_j3d[2][index1]).float()
        pred_pelvis = (pred_j3ds1[[2], :] + pred_j3ds1[[3], :]) / 2.0
        target_pelvis = (target_j3ds1[[2], :] + target_j3ds1[ [3], :]) / 2.0
        pred_j3ds1 -= pred_pelvis
        target_j3ds1 -= target_pelvis
        errors3 = torch.sqrt(((pred_j3ds1 - target_j3ds1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        pred_j3ds1 = torch.from_numpy(evaluation_accumulators_pred_j3d[3][index1]).float()
        target_j3ds1 = torch.from_numpy(evaluation_accumulators_target_j3d[3][index1]).float()
        pred_pelvis = (pred_j3ds1[[2], :] + pred_j3ds1[[3], :]) / 2.0
        target_pelvis = (target_j3ds1[[2], :] + target_j3ds1[[3], :]) / 2.0
        pred_j3ds1 -= pred_pelvis
        target_j3ds1 -= target_pelvis
        errors4 = torch.sqrt(((pred_j3ds1 - target_j3ds1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        errors_list=[float(errors1), float(errors2), float(errors3), float(errors4)]


        for index in range(camera_count):
            if(index==0):terget_list=[1,0,0,0]
            elif(index==1):terget_list=[0,1,0,0]
            elif(index==2):terget_list=[0,0,1,0]
            else:terget_list=[0,0,0,1]
            min_index=errors_list.index(min(errors_list))
            if(min_index==0):
                mode3_camera_index[vid_indices_con[index1]]=np.array(terget_list)
                errors_list[min_index]=1000
            elif(min_index==1):
                mode3_camera_index[vid_indices_con[index1+intervel_count]]=np.array(terget_list)
                errors_list[min_index] = 1000
            elif(min_index==2):
                mode3_camera_index[vid_indices_con[index1+2*intervel_count]]=np.array(terget_list)
                errors_list[min_index] = 1000
            else:
                mode3_camera_index[vid_indices_con[index1+3*intervel_count]]=np.array(terget_list)
                errors_list[min_index] = 1000

    save_file1 = pd.DataFrame(data=mode3_camera_index)
    save_file1.to_csv('C:\\H36M\\video_dir\\iris_training3_H36M_S9_11.csv')





    # for index in range(camera_count):

