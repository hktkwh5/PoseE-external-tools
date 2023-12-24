import torch
from multiprocessing import Pool
import numpy as np
from lib.data_utils.img_utils import get_single_image_crop

def Conv_BatchSeqImage_s(inp,inp_left,inp_right,batch_index,seq_index,target_3D):
    retuen_target_3D = torch.zeros_like(target_3D)
    for index1 in range(len(inp)):
        for index2 in range(len(inp[0])):
            inp[index1][index2] = inp[batch_index][seq_index]
            inp_left[index1][index2] = inp_left[batch_index][seq_index]
            inp_right[index1][index2] = inp_right[batch_index][seq_index]
            retuen_target_3D[index1][index2] = target_3D[batch_index][seq_index]
    return inp,inp_left,inp_right,retuen_target_3D

def Conv_BatchSeqImage_sTranslation(spin_model,inp,batch_index,seq_index,target_3D,image_name,bbox):
    retuen_target_3D = torch.zeros_like(target_3D)
    inp_con = torch.zeros_like(inp)
    video = torch.zeros(inp.shape[0]*inp.shape[1],3,224,224)
    count = 0
    len_trans = np.sqrt(len(video))
    center_batch,center_seq=0,0
    for index1 in range(len(inp)):
        for index2 in range(len(inp[0])):
            image_name[index1][index2]='F'+image_name[batch_index][seq_index][1:]
            col_index = int(count % len_trans)
            row_index = int(count / len_trans)
            x_index = 128-int((len_trans/2))+row_index
            y_index = 128-int((len_trans/2))+col_index
            if(x_index==128 and y_index==128):
                center_batch, center_seq = index1, index2
            video[count] = get_single_image_crop(image_name[index1][index2], [x_index, y_index, 256, 256], scale=1.0) # 这个地方可以试着将大图转换为目标图
            count+=1
    pred = spin_model.feature_extractor(video.cuda())
    count = 0
    for index1 in range(len(inp)):
        for index2 in range(len(inp[0])):
            # inp[index1][index2] = pred[count]
            inp_con[index1][index2] = pred[count]
            count += 1
            retuen_target_3D[index1][index2] = target_3D[batch_index][seq_index]
    return inp_con,retuen_target_3D,center_batch,center_seq

def Conv_BatchSeqImage_sInterpolation(spin_model,inp,batch_index,seq_index,target_3D,image_name,Tra_GroSam_3DTar,Tra_GroSam_Imgn,center_batch,nnn):
    retuen_target_3D = torch.zeros_like(target_3D)
    inp_con = torch.zeros_like(inp)
    video = torch.zeros(inp.shape[0] * inp.shape[1], 3, 224, 224)
    count = 0
    len_trans = np.sqrt(len(video))
    center_batch, center_seq = center_batch,4
    origin_batch, origin_seq,origin_count = 0, 0, 0

    number_SAME = center_batch*center_seq
    count_SAME = 0
    batch_index_list=[]
    seq_index_list = []
    count_index_list=[]
    one = number_SAME/nnn
    one_same = 0
    for index1 in range(len(inp)):
        if(index1%int(len(inp)/center_batch)==0):
            batch_index_list.append(index1)
            seq_index_list.append([])
            for index2 in range(len(inp[0])):
                if(index2%int(len(inp[0])/center_seq)==0):
                    if(count_SAME % nnn == 0):
                        if ((128-int(number_SAME/2)+one_same*nnn) == 128):
                            origin_batch, origin_seq = index1, index2
                            origin_count = count
                        Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[batch_index][seq_index][1:]
                        count_index_list.append(count)
                        video[count] = get_single_image_crop(Tra_GroSam_Imgn[index1][index2], [128, 128-int(one/2)+one_same, 256, 256],scale=1.0)  # 这个地方可以试着将大图转换为目标图
                        retuen_target_3D[index1][index2] = target_3D[batch_index][seq_index]
                        seq_index_list[len(seq_index_list)-1].append(index2)
                        count += 1
                        count_SAME += 1
                    elif (count_SAME % nnn == (nnn - 1)):
                        if(int(count_SAME % nnn) != batch_index):
                            Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[int(count_SAME % nnn)][seq_index][1:]
                        else:
                            if(batch_index!=0 or batch_index!=len(image_name)):
                                Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[0][seq_index][1:]
                            else:
                                Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[int(len(image_name)/2)][seq_index][1:]
                        video[count] = get_single_image_crop(Tra_GroSam_Imgn[index1][index2],
                                                             [128, 128 - int(one / 2) + one_same, 256, 256],
                                                             scale=1.0)
                        retuen_target_3D[index1][index2] = target_3D[batch_index][seq_index]
                        count += 1
                        count_SAME += 1
                        one_same+=1
                    else:
                        if (int(count_SAME % nnn) != batch_index):
                            Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[int(count_SAME % nnn)][
                                                                                        seq_index][1:]
                        else:
                            if (batch_index != 0 or batch_index != len(image_name)):
                                Tra_GroSam_Imgn[index1][index2] = 'F' + image_name[0][seq_index][1:]
                            else:
                                Tra_GroSam_Imgn[index1][index2] = 'F' + \
                                                                                  image_name[int(len(image_name) / 2)][
                                                                                      seq_index][1:]
                        video[count] = get_single_image_crop(Tra_GroSam_Imgn[index1][index2],
                                                             [128, 128 - int(one / 2) + one_same, 256, 256],
                                                             scale=1.0)
                        retuen_target_3D[index1][index2] = target_3D[batch_index][seq_index]
                        count += 1
                        count_SAME += 1
                else:
                    Tra_GroSam_Imgn[index1][index2] = 'F' + Tra_GroSam_Imgn[index1][index2][1:]
                    video[count] = get_single_image_crop(Tra_GroSam_Imgn[index1][index2], [128, 128, 256, 256],
                                                         scale=1.0)
                    retuen_target_3D[index1][index2] = Tra_GroSam_3DTar[index1][index2]
                    count += 1
        else:
            for index2 in range(len(inp[0])):
                Tra_GroSam_Imgn[index1][index2] = 'F' + Tra_GroSam_Imgn[index1][index2][1:]
                video[count] = get_single_image_crop(Tra_GroSam_Imgn[index1][index2], [128, 128, 256, 256],
                                                     scale=1.0)
                retuen_target_3D[index1][index2] = Tra_GroSam_3DTar[index1][index2]
                count += 1
    pred = spin_model.feature_extractor(video.cuda())
    count = 0
    for index1 in range(len(inp)):
        for index2 in range(len(inp[0])):
            inp_con[index1][index2] = pred[count]
            count += 1
    return inp_con, retuen_target_3D, batch_index_list,seq_index_list,count_index_list,origin_batch, origin_seq, origin_count

def generate_offset_matrix(distance):
    distance_sort = np.array(distance.cpu())

    for index in range(len(distance_sort[0])):
        distance_sort_index_x = np.argsort(distance_sort[:,index,0])
        distance_sort_index_y = np.argsort(distance_sort[:, index, 1])
        distance_sort_index_z = np.argsort(distance_sort[:, index, 2])
        distance_sort[:, index,0] = distance_sort[distance_sort_index_x, index,0]
        distance_sort[:, index,1] = distance_sort[distance_sort_index_y, index, 1]
        distance_sort[:, index,2] = distance_sort[distance_sort_index_z, index, 2]
    #寻找50%的样本中心点
    distance_sort_correct_50 = distance_sort[int(len(distance_sort)*0.5)]
    distance_sort_correct_70 = distance_sort[int(len(distance_sort) * 0.7)]
    distance_sort_correct_30 = distance_sort[int(len(distance_sort) * 0.3)]
    return distance_sort_correct_50,distance_sort_correct_70,distance_sort_correct_30

def generate_offset_justice_matrix(distance,count_index_list,origin_count):
    distance = distance[count_index_list]
    origin_sort_index = count_index_list.index(origin_count)
    distance_sort = np.array(distance.cpu())
    correct_matrix_50 = np.zeros((len(distance_sort[0]),len(distance_sort[0][0])))
    correct_matrix_30 = np.zeros((len(distance_sort[0]), len(distance_sort[0][0])))
    correct_matrix_70 = np.zeros((len(distance_sort[0]), len(distance_sort[0][0])))
    pool = Pool(processes=10)
    results = []
    index14_3 = []
    #在这里实现一个多进程计算，求出每个关节点以及xyz轴的纠正判断矩阵
    for index1 in range(len(distance_sort[0])):
        for index2 in range(len(distance_sort[0][0])):
            result = pool.apply_async(multi_generate_offset_justice_matrix, (distance_sort[:,index1,index2],origin_sort_index))
            results.append(result.get()[0])
            index14_3.append(result.get()[1])
    pool.close()
    pool.join()
    count=0
    index14_3_matrix=np.zeros_like(correct_matrix_50)
    for index1 in range(len(distance_sort[0])):
        for index2 in range(len(distance_sort[0][0])):
            # if(index2 == 1):
            if(results[count] == 1):
                correct_matrix_30[index1][index2]=1
            if(results[count] == 2):
                correct_matrix_50[index1][index2]=1
            if (results[count] == 3):
                correct_matrix_70[index1][index2]=1
            index14_3_matrix[index1][index2] = index14_3[count]
            count += 1

    return correct_matrix_30,correct_matrix_50,correct_matrix_70,index14_3_matrix

def multi_generate_offset_justice_matrix(origin_matrix,origin_sort_index):
    origin_matrix_sort_index = np.argsort(origin_matrix)
    origin_sorted_index = origin_matrix_sort_index.tolist().index(origin_sort_index)
    origin_matrix_sort = origin_matrix[origin_matrix_sort_index]
    number_smooth,number_rise,number_down = 30,10,10 #30,50,50
    grad_limit_rise_max = 0.02
    grad_limit_smooth,grad_limit_rise,grad_limit_down, grad_limit_rise_sum,grad_limit_down_sum = 0.02,10,0,0,0
    true_list_smooth, true_list_rise, true_list_down= [], [], []
    return_value=0
    count,count_max,true_list_smooth_=0,0,[]
    for index1 in range(0,len(origin_matrix_sort)-number_smooth):
        true_list_smooth_.append(index1)
        count += 1
        for index2 in range(index1+1,len(origin_matrix_sort)):
            if((origin_matrix_sort[index2]-origin_matrix_sort[index1])<=grad_limit_smooth):
                true_list_smooth_.append(index2)
                count+=1
            else:
                if(len(true_list_smooth_)<number_smooth):
                    true_list_smooth_=[]
                    count =0
                    break
                else:
                    if(count>count_max):
                        true_list_smooth = true_list_smooth_
                        true_list_smooth_ = []
                        count_max = count
                        count=0
    true_list_rise, true_list_rise_ = [], []
    count, count_max,grad_limit_rise_sum_max = 0, 0,0
    for index1 in range(0,len(origin_matrix_sort)-number_rise):
        grad_limit_rise=10
        grad_limit_rise_sum=0
        true_list_rise_.append(index1)
        count += 1
        for index2 in range(index1+1,len(origin_matrix_sort)):

            if((origin_matrix_sort[index2]-origin_matrix_sort[index2-1])<grad_limit_rise and (origin_matrix_sort[index2]-origin_matrix_sort[index2-1])>0):
                true_list_rise_.append(index2)
                count += 1
                grad_limit_rise_sum += origin_matrix_sort[index2]-origin_matrix_sort[index2-1]
                grad_limit_rise = origin_matrix_sort[index2]-origin_matrix_sort[index2-1]
            else:
                if (count > count_max or grad_limit_rise_sum>grad_limit_rise_sum_max):
                    grad_limit_rise_sum_max = grad_limit_rise_sum
                    true_list_rise = true_list_rise_
                    true_list_rise_ = []
                    count_max = count
                    count = 0
                break
        if(len(true_list_rise)>=number_rise and grad_limit_rise_sum>=grad_limit_rise_max):
            break
    if (len(true_list_rise) > number_rise or grad_limit_rise_sum < grad_limit_rise_max):# or count_limit_rise>=int(number_rise*0.5)
        true_list_rise=[]

    true_list_down, true_list_down_ = [], []
    count, count_max,grad_limit_down_sum_max = 0,0,0
    for index1 in range(0, len(origin_matrix_sort) - number_down):
        true_list_down = []
        grad_limit_down = 0
        grad_limit_down_sum = 0
        true_list_down.append(index1)
        count += 1
        for index2 in range(index1 + 1, len(origin_matrix_sort)):

            if ((origin_matrix_sort[index2] - origin_matrix_sort[index2 - 1]) > grad_limit_down):
                true_list_down.append(index2)
                count += 1
                grad_limit_down_sum += origin_matrix_sort[index2] - origin_matrix_sort[index2-1]
                grad_limit_down = origin_matrix_sort[index2] - origin_matrix_sort[index2-1]
            else:
                if (count > count_max or grad_limit_down_sum>grad_limit_down_sum_max):
                    true_list_down = true_list_down_
                    true_list_down_ = []
                    count_max = count
                    count = 0
                break
        if (len(true_list_down) >= number_down and grad_limit_down_sum <= grad_limit_rise_max):
            break
    if (len(true_list_down) > number_down or grad_limit_down_sum < grad_limit_rise_max):# or count_limit_down>=int(number_down*0.5)
        true_list_down = []

    if(len(true_list_smooth)>=number_smooth):
        if (true_list_smooth[-1] <= len(origin_matrix_sort) * 0.5):
            if(origin_sorted_index in true_list_smooth):
                return_value = 1
            elif(origin_sorted_index in true_list_rise):
                return_value = 1
            elif (origin_sorted_index in true_list_down):
                return_value = 3
            elif(origin_sorted_index>true_list_smooth[len(true_list_smooth)-1]):
                return_value = 2
        elif (true_list_smooth[0] >= len(origin_matrix_sort) * 0.5):
            if (origin_sorted_index in true_list_smooth):
                return_value = 3
            elif (origin_sorted_index in true_list_rise):
                return_value = 1
            elif (origin_sorted_index in true_list_down):
                return_value = 3
            elif (origin_sorted_index < true_list_smooth[0]):
                return_value = 2
        else:
            if (origin_sorted_index in true_list_smooth):
                return_value = 2
            elif (origin_sorted_index in true_list_rise):
                return_value = 1
            elif (origin_sorted_index in true_list_down):
                return_value = 3
            else:
                return_value = 0
    else:
        if (origin_sorted_index in true_list_rise):
            return_value = 1
        elif (origin_sorted_index in true_list_down):
            return_value = 3
        else:
            return_value = 0

    if(len(true_list_smooth)==0):
        true_list_smooth.append(0)

    return return_value,origin_matrix_sort_index[origin_sorted_index]