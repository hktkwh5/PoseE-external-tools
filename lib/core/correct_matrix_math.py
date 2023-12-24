import torch
import numpy as np
from lib.core.Conv_BatchSeqImage_s import generate_offset_matrix,generate_offset_justice_matrix

def Curr_matrix_toTar(istrain, center_batch, center_seq,preds_con_singe_tran,axis_joint_filter,target=None,preds=None,count_index_list=[],origin_count=0,aaa_value=1):
    if(istrain):
        preds_pelvis = (preds[:,:,[2], :] + preds[:,:,[3], :]) / 2.0
        preds_con_singe = preds - preds_pelvis
        target_pelvis = (target[:, :, [2], :] + target[:, :, [3], :]) / 2.0
        target_con_singe = target - target_pelvis
        distance = preds_con_singe - target_con_singe
        distance = distance.reshape(distance.size(0) * distance.size(1), distance.size(2),distance.size(3))
        distance_min = torch.abs(distance)
        distance_min = torch.sum(distance_min,2)
        distance_min = torch.sum(distance_min, 1)
        distance_min = distance_min[count_index_list]
        min_mpjpe_index = torch.argmin(distance_min,0)
        preds_con_singe_sort = preds_con_singe.reshape(preds_con_singe.size(0) * preds_con_singe.size(1), preds_con_singe.size(2), preds_con_singe.size(3))
        preds_con = preds.reshape(preds.size(0) * preds.size(1), preds.size(2),preds.size(3))
        joint_id = 3

        distance_sort_x = distance[:, joint_id, 0].reshape(1, len(distance[:, joint_id, 0]))
        distance_sort_x = np.array(distance_sort_x.cpu())
        DisSort_Index_x = np.argsort(distance_sort_x[0])
        DisSort_Index_x_same=[]
        DisSort_Index_x_S2 = []
        for index1 in range(len(count_index_list)):
            DisSort_Index_x_same.append(np.where(DisSort_Index_x == count_index_list[index1])[0])
        for index1 in range(len(DisSort_Index_x)):
            if(DisSort_Index_x[index1] not in count_index_list):
                DisSort_Index_x_S2.append(DisSort_Index_x[index1])
        DisSort_Index_x_S2 = np.array(DisSort_Index_x_S2)

        SamGro_OffsetMat_50,SamGro_OffsetMat_70,SamGro_OffsetMat_30=generate_offset_matrix(distance[DisSort_Index_x_S2])
        SamGro_OffsetJusticeMat_30, SamGro_OffsetJusticeMat_50,SamGro_OffsetJusticeMat_70,index14_3_matrix = generate_offset_justice_matrix(preds_con_singe_sort,count_index_list,origin_count)
        SamGro_OffsetMat_30_ = SamGro_OffsetMat_30 * SamGro_OffsetJusticeMat_30
        SamGro_OffsetMat_50_ = SamGro_OffsetMat_50 * SamGro_OffsetJusticeMat_50
        SamGro_OffsetMat_70_ = SamGro_OffsetMat_70 * SamGro_OffsetJusticeMat_70
        preds_con_origin = preds_con[count_index_list]

        OriMatrix_chin=np.zeros((len(preds_con_origin[0]),len(preds_con_origin[0][0])))
        for index1 in range(len(preds_con_origin[0])):
            for index2 in range(len(preds_con_origin[0][0])):
                OriMatrix_chin[index1][index2]=preds_con_origin[int(index14_3_matrix[index1][index2])][index1][index2]

        aaa=aaa_value
        OriMatrix_chin2 = OriMatrix_chin - SamGro_OffsetMat_30_*axis_joint_filter*aaa - SamGro_OffsetMat_50_*axis_joint_filter*aaa - SamGro_OffsetMat_70_*axis_joint_filter*aaa

        SamGro_OffsetJusticeMat_30, SamGro_OffsetJusticeMat_50, SamGro_OffsetJusticeMat_70, index14_3_matrix = generate_offset_justice_matrix(
            preds_con_singe_sort, count_index_list, count_index_list[min_mpjpe_index])
        SamGro_OffsetMat_30_ = SamGro_OffsetMat_30 * SamGro_OffsetJusticeMat_30
        SamGro_OffsetMat_50_ = SamGro_OffsetMat_50 * SamGro_OffsetJusticeMat_50
        SamGro_OffsetMat_70_ = SamGro_OffsetMat_70 * SamGro_OffsetJusticeMat_70

        OriMatrix_chin_mMP=np.zeros((len(preds_con_origin[0]),len(preds_con_origin[0][0])))
        for index1 in range(len(preds_con_origin[0])):
            for index2 in range(len(preds_con_origin[0][0])):
                OriMatrix_chin_mMP[index1][index2] = preds_con_origin[int(index14_3_matrix[index1][index2])][index1][
                    index2]
        OriMatrix_chin2_mMP = OriMatrix_chin_mMP - SamGro_OffsetMat_30_*axis_joint_filter*aaa - SamGro_OffsetMat_50_*axis_joint_filter*aaa - SamGro_OffsetMat_70_*axis_joint_filter*aaa

        distance_sort_y = distance[:, joint_id, 1].reshape(1, len(distance[:, joint_id, 1]))
        distance_sort_y = np.array(distance_sort_y.cpu())
        distance_sort_index_y = np.argsort(distance_sort_y[0])
        distance_sort_index_y_same = []
        distance_sort_index_y_same_2 = []
        for index1 in range(len(count_index_list)):
            distance_sort_index_y_same.append(np.where(distance_sort_index_y == count_index_list[index1])[0])
        for index1 in range(len(distance_sort_index_y)):
            if(distance_sort_index_y[index1] not in count_index_list):
                distance_sort_index_y_same_2.append(distance_sort_index_y[index1])
        distance_sort_z = distance[:, joint_id, 2].reshape(1, len(distance[:, joint_id, 2]))
        distance_sort_z = np.array(distance_sort_z.cpu())
        distance_sort_index_z = np.argsort(distance_sort_z[0])
        distance_sort_index_z_same = []
        distance_sort_index_z_same_2 = []
        for index1 in range(len(count_index_list)):
            distance_sort_index_z_same.append(np.where(distance_sort_index_z == count_index_list[index1])[0])
        for index1 in range(len(distance_sort_index_z)):
            if(distance_sort_index_z[index1] not in count_index_list):
                distance_sort_index_z_same_2.append(distance_sort_index_z[index1])

    else:
        print()

    return preds,OriMatrix_chin2,OriMatrix_chin,OriMatrix_chin2_mMP,OriMatrix_chin_mMP #,OriMatrix_chin_3