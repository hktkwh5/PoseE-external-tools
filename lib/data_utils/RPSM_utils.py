import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def camera_to_world_frame(x, R, T):
    """
    Args
        x: Nx3 points in camera coordinates#这里输入的x其实是相机坐标
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: Nx3 points in world coordinates
    """
    devices = torch.device('cuda')
    R = torch.tensor(torch.as_tensor(R, device='cuda'), dtype=torch.float32)
    T = torch.tensor(torch.as_tensor(T, device='cuda'), dtype=torch.float32)
    x = torch.tensor(torch.as_tensor(x, device='cuda'), dtype=torch.float32)
    xcam = torch.mm(torch.t(R), torch.t(x - T))  # 原本是xcam = torch.mm(torch.t(R), torch.t(x))
    # xcam = torch.t(xcam) + T  # rotate and translate
    return torch.t(xcam)


def compute_limb_length(body, pose):
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx].cpu() - pose[child].cpu())
            limb_length[(idx, child)] = length
    return limb_length

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

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
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform_pts_cuda(pts, t):
    npts = pts.shape[0]
    pts_homo = torch.cat([pts, torch.ones(npts, 1, device=pts.device)], dim=1)
    out = torch.mm(t, torch.t(pts_homo))
    return torch.t(out[:2, :])

def compute_grid(boxSize, boxCenter, nBins, device=None):
    grid1D = torch.linspace(-boxSize / 2, boxSize / 2, nBins, device=device)

    gridx, gridy, gridz = torch.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],
    )
    gridx = gridx.contiguous().view(-1, 1)
    gridy = gridy.contiguous().view(-1, 1)
    gridz = gridz.contiguous().view(-1, 1)
    grid = torch.cat([gridx, gridy, gridz], dim=1)
    return grid

def unfold_camera_param(camera, device=None):
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=torch.device('cuda'))
    T = torch.as_tensor(camera['T'], dtype=torch.float, device=torch.device('cuda'))
    f = torch.as_tensor(
        0.5 * (camera['fx'] + camera['fy']),
        dtype=torch.float,
        device=device)
    c = torch.as_tensor(
        [[camera['cx']], [camera['cy']]],
        dtype=torch.float,
        device=device)
    return R, T, f, c #, k, p

def project_point_radial(x, R, T, f, c):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = torch.mm(R, torch.t(x))
    xcam = torch.t(xcam) + T
    xcam = torch.t(xcam)
    y = xcam[:2] / xcam[2]
    ypixel = (f * y) + c
    return torch.t(ypixel)


def project_pose(x, camera):
    R, T, f, c = unfold_camera_param(camera, device=torch.device('cuda'))
    return project_point_radial(x, R, T, f, c)

def compute_unary_term(heatmap, grid, bbox2D, cam, imgSize):
    """
    Args:
        heatmap: array of size (n * k * h * w)
                -n: views,      -k: joints
                -h: height,     -w: width

        grid: k lists of ndarrays of size (nbins * 3)
                -k: joints; 1 when the grid is shared in PSM
                -nbins: bins in the grid

        bbox2D: bounding box on which heatmap is computed

    Returns:
        unary_of_all_joints: a list of ndarray of size nbins
    """
    device = heatmap.device
    share_grid = len(grid) == 1
    n, k = heatmap.shape[0], heatmap.shape[1]
    xyz=heatmap.shape[2]

    all_unary = {}
    for v in range(n):
        center = bbox2D[v]['center']
        scale = bbox2D[v]['scale']
        trans = torch.as_tensor(
            get_affine_transform(center, scale, 0, imgSize),
            dtype=torch.float,
            device=device)

        for j in range(k):
            grid_id = 0 if len(grid) == 1 else j
            nbins = grid[grid_id].shape[0]

            if (share_grid and j == 0) or not share_grid:
                xy = project_pose(grid[grid_id], cam[v])
                xy = affine_transform_pts_cuda(xy, trans) * torch.tensor(
                    [64, 64], dtype=torch.float, device=device) / torch.tensor(
                        imgSize, dtype=torch.float, device=device)
                sample_grid = xy / torch.tensor(
                    [64 - 1, 64 - 1], dtype=torch.float,
                    device=device) * 2.0 - 1.0
                sample_grid = sample_grid.view(1, 1, nbins, 2)
            unary_per_view_joint = F.grid_sample(
                heatmap[v:v + 1, j:j + 1, :], sample_grid)
            if j in all_unary:
                all_unary[j] += unary_per_view_joint
            else:
                all_unary[j] = unary_per_view_joint

    all_unary_list = []
    for j in range(k):
        all_unary_list.append(all_unary[j].view(1, -1))
    return all_unary_list

def infer(unary, pairwise, body, config):
    """
    Args:
        unary: [list] unary terms of all joints
        pairwise: [list] pairwise terms of all edges
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """
    root_idx = config

    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level

    states_of_all_joints = {}
    for node in skeleton_sorted_by_level:
        children_state = []
        u = unary[node['idx']].clone()
        if len(node['children']) == 0:
            energy = u
            children_state = [[-1]] * energy.numel()
        else:
            for child in node['children']:
                pw = pairwise[(node['idx'], child)]
                ce = states_of_all_joints[child]['Energy']
                ce = ce.expand_as(pw)
                pwce = torch.mul(pw, ce)
                max_v, max_i = torch.max(pwce, dim=1)
                u = torch.mul(u, max_v)#这里u是1*8的
                children_state.append(max_i.detach().cpu().numpy())

            children_state = np.array(children_state).T

        res = {'Energy': u, 'State': children_state}
        states_of_all_joints[node['idx']] = res
    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy'].detach().cpu().numpy()
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])
    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]
        children_index = skeleton[joint_idx]['children']#
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])
    pose3d_as_cube_idx.sort()
    return pose3d_as_cube_idx

def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = torch.zeros(njoints, 3, device=grid[0].device)
    single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d

def recursive_infer(initpose, cams, heatmaps, boxes, img_size, heatmap_size,
                    body, limb_length, grid_size, nbins, tolerance, config):
    device = heatmaps.device
    njoints = initpose.shape[0]
    grids = []
    for i in range(njoints):
        grids.append(compute_grid(grid_size, initpose[i], nbins, device=device))
    unary = compute_unary_term(heatmaps, grids, boxes, cams, img_size)
    skeleton = body.skeleton
    pairwise = compute_pairwise(skeleton, limb_length, grids, tolerance)
    pose3d_cube = infer(unary, pairwise, body, config)
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)
    return pose3d

def compute_pairwise(skeleton, limb_length, grid, tolerance):

    pairwise = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            expect_length = limb_length[(current, child)]
            distance = pdist2(grid[current], grid[child]) + 1e-9
            pairwise[(current, child)] = (torch.abs(distance - expect_length) <
                                          tolerance).float()
    return pairwise

def pdist2(x, y):
    """
    Compute distance between each pair of row vectors in x and y

    Args:
        x: tensor of shape n*p
        y: tensor of shape m*p
    Returns:
        dist: tensor of shape n*m
    """
    p = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    xtile = torch.cat([x] * m, dim=1).view(-1, p)
    ytile = torch.cat([y] * n, dim=0)
    dist = torch.pairwise_distance(xtile, ytile)
    return dist.view(n, m)

def rpsm(cams, heatmaps, kw, config):
    """
    Args:
        cams : camera parameters for each view
        heatmaps: 2d pose heatmaps (n, k, h, w)
    Returns:
        pose3d: 3d pose
    """

    # all in this device
    device = torch.device('cuda')
    img_size = config.NETWORK.IMAGE_SIZE
    map_size = config.NETWORK.HEATMAP_SIZE
    grd_size = config.PICT_STRUCT.GRID_SIZE
    fst_nbins = config.PICT_STRUCT.FIRST_NBINS
    rec_nbins = config.PICT_STRUCT.RECUR_NBINS
    rec_depth = config.PICT_STRUCT.RECUR_DEPTH
    tolerance = config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE

    grid = compute_grid(grd_size, kw['center'], fst_nbins, device=device)
    unary = compute_unary_term(heatmaps, [grid], kw['boxes'], cams,
                               img_size)

    pose3d_as_cube_idx = infer(unary, kw['pairwise'], kw['body'], config)
    pose3d = get_loc_from_cube_idx([grid],pose3d_as_cube_idx)

    cur_grd_size = grd_size / fst_nbins
    for i in range(rec_depth):
        pose3d = recursive_infer(pose3d, cams, heatmaps, kw['boxes'], img_size,
                                 map_size, kw['body'], kw['limb_length'],
                                 cur_grd_size, rec_nbins, tolerance,
                                 config)
        cur_grd_size = cur_grd_size / rec_nbins

    return pose3d

def double_view_sample_plot(evaluation_accumulators_pred_j3d,evaluation_accumulators_target_j3d,evaluation_accumulators_camera_para,evaluation_accumulators_center,evaluation_accumulators_scaler):
    pred_j3ds_group1, target_j3ds_group1, camera_para_group1, box_center_group1, box_scaler_group1 = evaluation_accumulators_pred_j3d[1], evaluation_accumulators_target_j3d[1], evaluation_accumulators_camera_para[1], evaluation_accumulators_center[1], evaluation_accumulators_scaler[1]
    pred_j3ds_group2, target_j3ds_group2, camera_para_group2, box_center_group2, box_scaler_group2 = evaluation_accumulators_pred_j3d[2], evaluation_accumulators_target_j3d[2], evaluation_accumulators_camera_para[2], evaluation_accumulators_center[2], evaluation_accumulators_scaler[2]
    poses_group1 = []
    poses_group2 = []
    for id_utem in range(len(target_j3ds_group1)):
        poses_group1.append(camera_to_world_frame(pred_j3ds_group1[id_utem], camera_para_group1[id_utem][0]['R'],camera_para_group1[id_utem][0]['T']))
        poses_group2.append(camera_to_world_frame(pred_j3ds_group2[id_utem], camera_para_group2[id_utem][0]['R'],camera_para_group2[id_utem][0]['T']))

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group1[0][13, 0],
                                                                                 poses_group1[0][12, 0]], [
                                                                                    poses_group1[0][12, 0],
                                                                                    poses_group1[0][9, 0]], [
                                                                                    poses_group1[0][12, 0],
                                                                                    poses_group1[0][8, 0]], [
                                                                                    poses_group1[0][9, 0],
                                                                                    poses_group1[0][10, 0]], [
                                                                                    poses_group1[0][10, 0],
                                                                                    poses_group1[0][11, 0]], [
                                                                                    poses_group1[0][8, 0],
                                                                                    poses_group1[0][7, 0]], [
                                                                                    poses_group1[0][7, 0],
                                                                                    poses_group1[0][6, 0]], [
                                                                                    poses_group1[0][8, 0],
                                                                                    poses_group1[0][2, 0]], [
                                                                                    poses_group1[0][9, 0],
                                                                                    poses_group1[0][3, 0]], [
                                                                                    poses_group1[0][2, 0],
                                                                                    poses_group1[0][3, 0]], [
                                                                                    poses_group1[0][2, 0],
                                                                                    poses_group1[0][1, 0]], [
                                                                                    poses_group1[0][1, 0],
                                                                                    poses_group1[0][0, 0]], [
                                                                                    poses_group1[0][3, 0],
                                                                                    poses_group1[0][4, 0]], [
                                                                                    poses_group1[0][4, 0],
                                                                                    poses_group1[0][5, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group1[0][13, 1],
                                                                                 poses_group1[0][12, 1]], [
                                                                                    poses_group1[0][12, 1],
                                                                                    poses_group1[0][9, 1]], [
                                                                                    poses_group1[0][12, 1],
                                                                                    poses_group1[0][8, 1]], [
                                                                                    poses_group1[0][9, 1],
                                                                                    poses_group1[0][10, 1]], [
                                                                                    poses_group1[0][10, 1],
                                                                                    poses_group1[0][11, 1]], [
                                                                                    poses_group1[0][8, 1],
                                                                                    poses_group1[0][7, 1]], [
                                                                                    poses_group1[0][7, 1],
                                                                                    poses_group1[0][6, 1]], [
                                                                                    poses_group1[0][8, 1],
                                                                                    poses_group1[0][2, 1]], [
                                                                                    poses_group1[0][9, 1],
                                                                                    poses_group1[0][3, 1]], [
                                                                                    poses_group1[0][2, 1],
                                                                                    poses_group1[0][3, 1]], [
                                                                                    poses_group1[0][2, 1],
                                                                                    poses_group1[0][1, 1]], [
                                                                                    poses_group1[0][1, 1],
                                                                                    poses_group1[0][0, 1]], [
                                                                                    poses_group1[0][3, 1],
                                                                                    poses_group1[0][4, 1]], [
                                                                                    poses_group1[0][4, 1],
                                                                                    poses_group1[0][5, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group1[0][13, 2],
                                                                                 poses_group1[0][12, 2]], [
                                                                                    poses_group1[0][12, 2],
                                                                                    poses_group1[0][9, 2]], [
                                                                                    poses_group1[0][12, 2],
                                                                                    poses_group1[0][8, 2]], [
                                                                                    poses_group1[0][9, 2],
                                                                                    poses_group1[0][10, 2]], [
                                                                                    poses_group1[0][10, 2],
                                                                                    poses_group1[0][11, 2]], [
                                                                                    poses_group1[0][8, 2],
                                                                                    poses_group1[0][7, 2]], [
                                                                                    poses_group1[0][7, 2],
                                                                                    poses_group1[0][6, 2]], [
                                                                                    poses_group1[0][8, 2],
                                                                                    poses_group1[0][2, 2]], [
                                                                                    poses_group1[0][9, 2],
                                                                                    poses_group1[0][3, 2]], [
                                                                                    poses_group1[0][2, 2],
                                                                                    poses_group1[0][3, 2]], [
                                                                                    poses_group1[0][2, 2],
                                                                                    poses_group1[0][1, 2]], [
                                                                                    poses_group1[0][1, 2],
                                                                                    poses_group1[0][0, 2]], [
                                                                                    poses_group1[0][3, 2],
                                                                                    poses_group1[0][4, 2]], [
                                                                                    poses_group1[0][4, 2],
                                                                                    poses_group1[0][5, 2]]
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.plot(xs1, ys1, zs1, c='red', marker='o')
    ax.plot(xs2, ys2, zs2, c='red', marker='o')
    ax.plot(xs3, ys3, zs3, c='red', marker='o')
    ax.plot(xs4, ys4, zs4, c='red', marker='o')
    ax.plot(xs5, ys5, zs5, c='red', marker='o')
    ax.plot(xs6, ys6, zs6, c='red', marker='o')
    ax.plot(xs7, ys7, zs7, c='red', marker='o')
    ax.plot(xs8, ys8, zs8, c='red', marker='o')
    ax.plot(xs9, ys9, zs9, c='red', marker='o')
    ax.plot(xs10, ys10, zs10, c='red', marker='o')
    ax.plot(xs11, ys11, zs11, c='red', marker='o')
    ax.plot(xs12, ys12, zs12, c='red', marker='o')
    ax.plot(xs13, ys13, zs13, c='red', marker='o')
    ax.plot(xs14, ys14, zs14, c='red', marker='o')
    plt.show()

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group2[0][13, 0],
                                                                                 poses_group2[0][12, 0]], [
                                                                                    poses_group2[0][12, 0],
                                                                                    poses_group2[0][9, 0]], [
                                                                                    poses_group2[0][12, 0],
                                                                                    poses_group2[0][8, 0]], [
                                                                                    poses_group2[0][9, 0],
                                                                                    poses_group2[0][10, 0]], [
                                                                                    poses_group2[0][10, 0],
                                                                                    poses_group2[0][11, 0]], [
                                                                                    poses_group2[0][8, 0],
                                                                                    poses_group2[0][7, 0]], [
                                                                                    poses_group2[0][7, 0],
                                                                                    poses_group2[0][6, 0]], [
                                                                                    poses_group2[0][8, 0],
                                                                                    poses_group2[0][2, 0]], [
                                                                                    poses_group2[0][9, 0],
                                                                                    poses_group2[0][3, 0]], [
                                                                                    poses_group2[0][2, 0],
                                                                                    poses_group2[0][3, 0]], [
                                                                                    poses_group2[0][2, 0],
                                                                                    poses_group2[0][1, 0]], [
                                                                                    poses_group2[0][1, 0],
                                                                                    poses_group2[0][0, 0]], [
                                                                                    poses_group2[0][3, 0],
                                                                                    poses_group2[0][4, 0]], [
                                                                                    poses_group2[0][4, 0],
                                                                                    poses_group2[0][5, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group2[0][13, 1],
                                                                                 poses_group2[0][12, 1]], [
                                                                                    poses_group2[0][12, 1],
                                                                                    poses_group2[0][9, 1]], [
                                                                                    poses_group2[0][12, 1],
                                                                                    poses_group2[0][8, 1]], [
                                                                                    poses_group2[0][9, 1],
                                                                                    poses_group2[0][10, 1]], [
                                                                                    poses_group2[0][10, 1],
                                                                                    poses_group2[0][11, 1]], [
                                                                                    poses_group2[0][8, 1],
                                                                                    poses_group2[0][7, 1]], [
                                                                                    poses_group2[0][7, 1],
                                                                                    poses_group2[0][6, 1]], [
                                                                                    poses_group2[0][8, 1],
                                                                                    poses_group2[0][2, 1]], [
                                                                                    poses_group2[0][9, 1],
                                                                                    poses_group2[0][3, 1]], [
                                                                                    poses_group2[0][2, 1],
                                                                                    poses_group2[0][3, 1]], [
                                                                                    poses_group2[0][2, 1],
                                                                                    poses_group2[0][1, 1]], [
                                                                                    poses_group2[0][1, 1],
                                                                                    poses_group2[0][0, 1]], [
                                                                                    poses_group2[0][3, 1],
                                                                                    poses_group2[0][4, 1]], [
                                                                                    poses_group2[0][4, 1],
                                                                                    poses_group2[0][5, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group2[0][13, 2],
                                                                                 poses_group2[0][12, 2]], [
                                                                                    poses_group2[0][12, 2],
                                                                                    poses_group2[0][9, 2]], [
                                                                                    poses_group2[0][12, 2],
                                                                                    poses_group2[0][8, 2]], [
                                                                                    poses_group2[0][9, 2],
                                                                                    poses_group2[0][10, 2]], [
                                                                                    poses_group2[0][10, 2],
                                                                                    poses_group2[0][11, 2]], [
                                                                                    poses_group2[0][8, 2],
                                                                                    poses_group2[0][7, 2]], [
                                                                                    poses_group2[0][7, 2],
                                                                                    poses_group2[0][6, 2]], [
                                                                                    poses_group2[0][8, 2],
                                                                                    poses_group2[0][2, 2]], [
                                                                                    poses_group2[0][9, 2],
                                                                                    poses_group2[0][3, 2]], [
                                                                                    poses_group2[0][2, 2],
                                                                                    poses_group2[0][3, 2]], [
                                                                                    poses_group2[0][2, 2],
                                                                                    poses_group2[0][1, 2]], [
                                                                                    poses_group2[0][1, 2],
                                                                                    poses_group2[0][0, 2]], [
                                                                                    poses_group2[0][3, 2],
                                                                                    poses_group2[0][4, 2]], [
                                                                                    poses_group2[0][4, 2],
                                                                                    poses_group2[0][5, 2]]
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.plot(xs1, ys1, zs1, c='red', marker='o')
    ax.plot(xs2, ys2, zs2, c='red', marker='o')
    ax.plot(xs3, ys3, zs3, c='red', marker='o')
    ax.plot(xs4, ys4, zs4, c='red', marker='o')
    ax.plot(xs5, ys5, zs5, c='red', marker='o')
    ax.plot(xs6, ys6, zs6, c='red', marker='o')
    ax.plot(xs7, ys7, zs7, c='red', marker='o')
    ax.plot(xs8, ys8, zs8, c='red', marker='o')
    ax.plot(xs9, ys9, zs9, c='red', marker='o')
    ax.plot(xs10, ys10, zs10, c='red', marker='o')
    ax.plot(xs11, ys11, zs11, c='red', marker='o')
    ax.plot(xs12, ys12, zs12, c='red', marker='o')
    ax.plot(xs13, ys13, zs13, c='red', marker='o')
    ax.plot(xs14, ys14, zs14, c='red', marker='o')
    plt.show()

def double_view_sample_plot_inner(poses_group1,poses_group2,target_group1):
    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group1[13, 0],
                                                                                 poses_group1[12, 0]], [
                                                                                    poses_group1[12, 0],
                                                                                    poses_group1[9, 0]], [
                                                                                    poses_group1[12, 0],
                                                                                    poses_group1[8, 0]], [
                                                                                    poses_group1[9, 0],
                                                                                    poses_group1[10, 0]], [
                                                                                    poses_group1[10, 0],
                                                                                    poses_group1[11, 0]], [
                                                                                    poses_group1[8, 0],
                                                                                    poses_group1[7, 0]], [
                                                                                    poses_group1[7, 0],
                                                                                    poses_group1[6, 0]], [
                                                                                    poses_group1[8, 0],
                                                                                    poses_group1[2, 0]], [
                                                                                    poses_group1[9, 0],
                                                                                    poses_group1[3, 0]], [
                                                                                    poses_group1[2, 0],
                                                                                    poses_group1[3, 0]], [
                                                                                    poses_group1[2, 0],
                                                                                    poses_group1[1, 0]], [
                                                                                    poses_group1[1, 0],
                                                                                    poses_group1[0, 0]], [
                                                                                    poses_group1[3, 0],
                                                                                    poses_group1[4, 0]], [
                                                                                    poses_group1[4, 0],
                                                                                    poses_group1[5, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group1[13, 1],
                                                                                 poses_group1[12, 1]], [
                                                                                    poses_group1[12, 1],
                                                                                    poses_group1[9, 1]], [
                                                                                    poses_group1[12, 1],
                                                                                    poses_group1[8, 1]], [
                                                                                    poses_group1[9, 1],
                                                                                    poses_group1[10, 1]], [
                                                                                    poses_group1[10, 1],
                                                                                    poses_group1[11, 1]], [
                                                                                    poses_group1[8, 1],
                                                                                    poses_group1[7, 1]], [
                                                                                    poses_group1[7, 1],
                                                                                    poses_group1[6, 1]], [
                                                                                    poses_group1[8, 1],
                                                                                    poses_group1[2, 1]], [
                                                                                    poses_group1[9, 1],
                                                                                    poses_group1[3, 1]], [
                                                                                    poses_group1[2, 1],
                                                                                    poses_group1[3, 1]], [
                                                                                    poses_group1[2, 1],
                                                                                    poses_group1[1, 1]], [
                                                                                    poses_group1[1, 1],
                                                                                    poses_group1[0, 1]], [
                                                                                    poses_group1[3, 1],
                                                                                    poses_group1[4, 1]], [
                                                                                    poses_group1[4, 1],
                                                                                    poses_group1[5, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group1[13, 2],
                                                                                 poses_group1[12, 2]], [
                                                                                    poses_group1[12, 2],
                                                                                    poses_group1[9, 2]], [
                                                                                    poses_group1[12, 2],
                                                                                    poses_group1[8, 2]], [
                                                                                    poses_group1[9, 2],
                                                                                    poses_group1[10, 2]], [
                                                                                    poses_group1[10, 2],
                                                                                    poses_group1[11, 2]], [
                                                                                    poses_group1[8, 2],
                                                                                    poses_group1[7, 2]], [
                                                                                    poses_group1[7, 2],
                                                                                    poses_group1[6, 2]], [
                                                                                    poses_group1[8, 2],
                                                                                    poses_group1[2, 2]], [
                                                                                    poses_group1[9, 2],
                                                                                    poses_group1[3, 2]], [
                                                                                    poses_group1[2, 2],
                                                                                    poses_group1[3, 2]], [
                                                                                    poses_group1[2, 2],
                                                                                    poses_group1[1, 2]], [
                                                                                    poses_group1[1, 2],
                                                                                    poses_group1[0, 2]], [
                                                                                    poses_group1[3, 2],
                                                                                    poses_group1[4, 2]], [
                                                                                    poses_group1[4, 2],
                                                                                    poses_group1[5, 2]]
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.plot(xs1, ys1, zs1, c='red', marker='o')
    ax.plot(xs2, ys2, zs2, c='red', marker='o')
    ax.plot(xs3, ys3, zs3, c='red', marker='o')
    ax.plot(xs4, ys4, zs4, c='red', marker='o')
    ax.plot(xs5, ys5, zs5, c='red', marker='o')
    ax.plot(xs6, ys6, zs6, c='red', marker='o')
    ax.plot(xs7, ys7, zs7, c='red', marker='o')
    ax.plot(xs8, ys8, zs8, c='red', marker='o')
    ax.plot(xs9, ys9, zs9, c='red', marker='o')
    ax.plot(xs10, ys10, zs10, c='red', marker='o')
    ax.plot(xs11, ys11, zs11, c='red', marker='o')
    ax.plot(xs12, ys12, zs12, c='red', marker='o')
    ax.plot(xs13, ys13, zs13, c='red', marker='o')
    ax.plot(xs14, ys14, zs14, c='red', marker='o')
    # plt.show()

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group2[13, 0],
                                                                                 poses_group2[12, 0]], [
                                                                                    poses_group2[12, 0],
                                                                                    poses_group2[9, 0]], [
                                                                                    poses_group2[12, 0],
                                                                                    poses_group2[8, 0]], [
                                                                                    poses_group2[9, 0],
                                                                                    poses_group2[10, 0]], [
                                                                                    poses_group2[10, 0],
                                                                                    poses_group2[11, 0]], [
                                                                                    poses_group2[8, 0],
                                                                                    poses_group2[7, 0]], [
                                                                                    poses_group2[7, 0],
                                                                                    poses_group2[6, 0]], [
                                                                                    poses_group2[8, 0],
                                                                                    poses_group2[2, 0]], [
                                                                                    poses_group2[9, 0],
                                                                                    poses_group2[3, 0]], [
                                                                                    poses_group2[2, 0],
                                                                                    poses_group2[3, 0]], [
                                                                                    poses_group2[2, 0],
                                                                                    poses_group2[1, 0]], [
                                                                                    poses_group2[1, 0],
                                                                                    poses_group2[0, 0]], [
                                                                                    poses_group2[3, 0],
                                                                                    poses_group2[4, 0]], [
                                                                                    poses_group2[4, 0],
                                                                                    poses_group2[5, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group2[13, 1],
                                                                                 poses_group2[12, 1]], [
                                                                                    poses_group2[12, 1],
                                                                                    poses_group2[9, 1]], [
                                                                                    poses_group2[12, 1],
                                                                                    poses_group2[8, 1]], [
                                                                                    poses_group2[9, 1],
                                                                                    poses_group2[10, 1]], [
                                                                                    poses_group2[10, 1],
                                                                                    poses_group2[11, 1]], [
                                                                                    poses_group2[8, 1],
                                                                                    poses_group2[7, 1]], [
                                                                                    poses_group2[7, 1],
                                                                                    poses_group2[6, 1]], [
                                                                                    poses_group2[8, 1],
                                                                                    poses_group2[2, 1]], [
                                                                                    poses_group2[9, 1],
                                                                                    poses_group2[3, 1]], [
                                                                                    poses_group2[2, 1],
                                                                                    poses_group2[3, 1]], [
                                                                                    poses_group2[2, 1],
                                                                                    poses_group2[1, 1]], [
                                                                                    poses_group2[1, 1],
                                                                                    poses_group2[0, 1]], [
                                                                                    poses_group2[3, 1],
                                                                                    poses_group2[4, 1]], [
                                                                                    poses_group2[4, 1],
                                                                                    poses_group2[5, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group2[13, 2],
                                                                                 poses_group2[12, 2]], [
                                                                                    poses_group2[12, 2],
                                                                                    poses_group2[9, 2]], [
                                                                                    poses_group2[12, 2],
                                                                                    poses_group2[8, 2]], [
                                                                                    poses_group2[9, 2],
                                                                                    poses_group2[10, 2]], [
                                                                                    poses_group2[10, 2],
                                                                                    poses_group2[11, 2]], [
                                                                                    poses_group2[8, 2],
                                                                                    poses_group2[7, 2]], [
                                                                                    poses_group2[7, 2],
                                                                                    poses_group2[6, 2]], [
                                                                                    poses_group2[8, 2],
                                                                                    poses_group2[2, 2]], [
                                                                                    poses_group2[9, 2],
                                                                                    poses_group2[3, 2]], [
                                                                                    poses_group2[2, 2],
                                                                                    poses_group2[3, 2]], [
                                                                                    poses_group2[2, 2],
                                                                                    poses_group2[1, 2]], [
                                                                                    poses_group2[1, 2],
                                                                                    poses_group2[0, 2]], [
                                                                                    poses_group2[3, 2],
                                                                                    poses_group2[4, 2]], [
                                                                                    poses_group2[4, 2],
                                                                                    poses_group2[5, 2]]



    ax.plot(xs1, ys1, zs1, c='blue', marker='o')
    ax.plot(xs2, ys2, zs2, c='blue', marker='o')
    ax.plot(xs3, ys3, zs3, c='blue', marker='o')
    ax.plot(xs4, ys4, zs4, c='blue', marker='o')
    ax.plot(xs5, ys5, zs5, c='blue', marker='o')
    ax.plot(xs6, ys6, zs6, c='blue', marker='o')
    ax.plot(xs7, ys7, zs7, c='blue', marker='o')
    ax.plot(xs8, ys8, zs8, c='blue', marker='o')
    ax.plot(xs9, ys9, zs9, c='blue', marker='o')
    ax.plot(xs10, ys10, zs10, c='blue', marker='o')
    ax.plot(xs11, ys11, zs11, c='blue', marker='o')
    ax.plot(xs12, ys12, zs12, c='blue', marker='o')
    ax.plot(xs13, ys13, zs13, c='blue', marker='o')
    ax.plot(xs14, ys14, zs14, c='blue', marker='o')

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [target_group1[13, 0],
                                                                                 target_group1[12, 0]], [
        target_group1[12, 0],
        target_group1[9, 0]], [
        target_group1[12, 0],
        target_group1[8, 0]], [
        target_group1[9, 0],
        target_group1[10, 0]], [
        target_group1[10, 0],
        target_group1[11, 0]], [
        target_group1[8, 0],
        target_group1[7, 0]], [
        target_group1[7, 0],
        target_group1[6, 0]], [
        target_group1[8, 0],
        target_group1[2, 0]], [
        target_group1[9, 0],
        target_group1[3, 0]], [
        target_group1[2, 0],
        target_group1[3, 0]], [
        target_group1[2, 0],
        target_group1[1, 0]], [
        target_group1[1, 0],
        target_group1[0, 0]], [
        target_group1[3, 0],
        target_group1[4, 0]], [
        target_group1[4, 0],
        target_group1[5, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [target_group1[13, 1],
                                                                                 target_group1[12, 1]], [
        target_group1[12, 1],
        target_group1[9, 1]], [
        target_group1[12, 1],
        target_group1[8, 1]], [
        target_group1[9, 1],
        target_group1[10, 1]], [
        target_group1[10, 1],
        target_group1[11, 1]], [
        target_group1[8, 1],
        target_group1[7, 1]], [
        target_group1[7, 1],
        target_group1[6, 1]], [
        target_group1[8, 1],
        target_group1[2, 1]], [
        target_group1[9, 1],
        target_group1[3, 1]], [
        target_group1[2, 1],
        target_group1[3, 1]], [
        target_group1[2, 1],
        target_group1[1, 1]], [
        target_group1[1, 1],
        target_group1[0, 1]], [
        target_group1[3, 1],
        target_group1[4, 1]], [
        target_group1[4, 1],
        target_group1[5, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [target_group1[13, 2],
                                                                                 target_group1[12, 2]], [
        target_group1[12, 2],
        target_group1[9, 2]], [
        target_group1[12, 2],
        target_group1[8, 2]], [
        target_group1[9, 2],
        target_group1[10, 2]], [
        target_group1[10, 2],
        target_group1[11, 2]], [
        target_group1[8, 2],
        target_group1[7, 2]], [
        target_group1[7, 2],
        target_group1[6, 2]], [
        target_group1[8, 2],
        target_group1[2, 2]], [
        target_group1[9, 2],
        target_group1[3, 2]], [
        target_group1[2, 2],
        target_group1[3, 2]], [
        target_group1[2, 2],
        target_group1[1, 2]], [
        target_group1[1, 2],
        target_group1[0, 2]], [
        target_group1[3, 2],
        target_group1[4, 2]], [
        target_group1[4, 2],
        target_group1[5, 2]]


    ax.plot(xs1, ys1, zs1, c='green', marker='o')
    ax.plot(xs2, ys2, zs2, c='green', marker='o')
    ax.plot(xs3, ys3, zs3, c='green', marker='o')
    ax.plot(xs4, ys4, zs4, c='green', marker='o')
    ax.plot(xs5, ys5, zs5, c='green', marker='o')
    ax.plot(xs6, ys6, zs6, c='green', marker='o')
    ax.plot(xs7, ys7, zs7, c='green', marker='o')
    ax.plot(xs8, ys8, zs8, c='green', marker='o')
    ax.plot(xs9, ys9, zs9, c='green', marker='o')
    ax.plot(xs10, ys10, zs10, c='green', marker='o')
    ax.plot(xs11, ys11, zs11, c='green', marker='o')
    ax.plot(xs12, ys12, zs12, c='green', marker='o')
    ax.plot(xs13, ys13, zs13, c='green', marker='o')
    ax.plot(xs14, ys14, zs14, c='green', marker='o')
    plt.show()

def double_view_sample_plot_inner_distance(poses_group1,poses_group2,target_group1):

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group1[13, 0],
                                                                                 target_group1[13, 0]], [
                                                                                    poses_group1[12, 0],
        target_group1[12, 0]], [
                                                                                    poses_group1[11, 0],
        target_group1[11, 0]], [
                                                                                    poses_group1[10, 0],
        target_group1[10, 0]], [
                                                                                    poses_group1[9, 0],
        target_group1[9, 0]], [
                                                                                    poses_group1[8, 0],
        target_group1[8, 0]], [
                                                                                    poses_group1[7, 0],
        target_group1[7, 0]], [
                                                                                    poses_group1[6, 0],
        target_group1[6, 0]], [
                                                                                    poses_group1[5, 0],
        target_group1[5, 0]], [
                                                                                    poses_group1[4, 0],
        target_group1[4, 0]], [
                                                                                    poses_group1[3, 0],
        target_group1[3, 0]], [
                                                                                    poses_group1[2, 0],
        target_group1[2, 0]], [
                                                                                    poses_group1[1, 0],
        target_group1[1, 0]], [
                                                                                    poses_group1[0, 0],
        target_group1[0, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group1[13, 1],
                                                                                 target_group1[13, 1]], [
                                                                                    poses_group1[12, 1],
        target_group1[12, 1]], [
                                                                                    poses_group1[11, 1],
        target_group1[11, 1]], [
                                                                                    poses_group1[10, 1],
        target_group1[10, 1]], [
                                                                                    poses_group1[9, 1],
        target_group1[9, 1]], [
                                                                                    poses_group1[8, 1],
        target_group1[8, 1]], [
                                                                                    poses_group1[7, 1],
        target_group1[7, 1]], [
                                                                                    poses_group1[6, 1],
        target_group1[6, 1]], [
                                                                                    poses_group1[5, 1],
        target_group1[5, 1]], [
                                                                                    poses_group1[4, 1],
        target_group1[4, 1]], [
                                                                                    poses_group1[3, 1],
        target_group1[3, 1]], [
                                                                                    poses_group1[2, 1],
        target_group1[2, 1]], [
                                                                                    poses_group1[1, 1],
        target_group1[1, 1]], [
                                                                                    poses_group1[0, 1],
        target_group1[0, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group1[13, 2],
                                                                                 target_group1[13, 2]], [
                                                                                    poses_group1[12, 2],
        target_group1[12, 2]], [
                                                                                    poses_group1[11, 2],
        target_group1[11, 2]], [
                                                                                    poses_group1[10, 2],
        target_group1[10, 2]], [
                                                                                    poses_group1[9, 2],
        target_group1[9, 2]], [
                                                                                    poses_group1[8, 2],
        target_group1[8, 2]], [
                                                                                    poses_group1[7, 2],
        target_group1[7, 2]], [
                                                                                    poses_group1[6, 2],
        target_group1[6, 2]], [
                                                                                    poses_group1[5, 2],
        target_group1[5, 2]], [
                                                                                    poses_group1[4, 2],
        target_group1[4, 2]], [
                                                                                    poses_group1[3, 2],
        target_group1[3, 2]], [
                                                                                    poses_group1[2, 2],
        target_group1[2, 2]], [
                                                                                    poses_group1[1, 2],
        target_group1[1, 2]], [
                                                                                    poses_group1[0, 2],
        target_group1[0, 2]]
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')

    ax.plot(xs1, ys1, zs1, c='red', marker='o',alpha=0.3)
    ax.plot(xs2, ys2, zs2, c='red', marker='o',alpha=0.3)
    ax.plot(xs3, ys3, zs3, c='red', marker='o',alpha=0.3)
    ax.plot(xs4, ys4, zs4, c='red', marker='o',alpha=0.3)
    ax.plot(xs5, ys5, zs5, c='red', marker='o',alpha=0.3)
    ax.plot(xs6, ys6, zs6, c='red', marker='o',alpha=0.3)
    ax.plot(xs7, ys7, zs7, c='red', marker='o',alpha=0.3)
    ax.plot(xs8, ys8, zs8, c='red', marker='o',alpha=0.3)
    ax.plot(xs9, ys9, zs9, c='red', marker='o',alpha=0.3)
    ax.plot(xs10, ys10, zs10, c='red', marker='o',alpha=0.3)
    ax.plot(xs11, ys11, zs11, c='red', marker='o',alpha=0.3)
    ax.plot(xs12, ys12, zs12, c='red', marker='o',alpha=0.3)
    ax.plot(xs13, ys13, zs13, c='red', marker='o',alpha=0.3)
    ax.plot(xs14, ys14, zs14, c='red', marker='o',alpha=0.3)
    # plt.show()

    xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14 = [poses_group2[13, 0],
                                                                                 target_group1[13, 0]], [
                                                                                    poses_group2[12, 0],
        target_group1[12, 0]], [
                                                                                    poses_group2[11, 0],
        target_group1[11, 0]], [
                                                                                    poses_group2[10, 0],
        target_group1[10, 0]], [
                                                                                    poses_group2[9, 0],
        target_group1[9, 0]], [
                                                                                    poses_group2[8, 0],
        target_group1[8, 0]], [
                                                                                    poses_group2[7, 0],
        target_group1[7, 0]], [
                                                                                    poses_group2[6, 0],
        target_group1[6, 0]], [
                                                                                    poses_group2[5, 0],
        target_group1[5, 0]], [
                                                                                    poses_group2[4, 0],
        target_group1[4, 0]], [
                                                                                    poses_group2[3, 0],
        target_group1[3, 0]], [
                                                                                    poses_group2[2, 0],
        target_group1[2, 0]], [
                                                                                    poses_group2[1, 0],
        target_group1[1, 0]], [
                                                                                    poses_group2[0, 0],
        target_group1[0, 0]]
    ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys10, ys11, ys12, ys13, ys14 = [poses_group2[13, 1],
                                                                                 target_group1[13, 1]], [
                                                                                    poses_group2[12, 1],
        target_group1[12, 1]], [
                                                                                    poses_group2[11, 1],
        target_group1[11, 1]], [
                                                                                    poses_group2[10, 1],
        target_group1[10, 1]], [
                                                                                    poses_group2[9, 1],
        target_group1[9, 1]], [
                                                                                    poses_group2[8, 1],
        target_group1[8, 1]], [
                                                                                    poses_group2[7, 1],
        target_group1[7, 1]], [
                                                                                    poses_group2[6, 1],
        target_group1[6, 1]], [
                                                                                    poses_group2[5, 1],
        target_group1[5, 1]], [
                                                                                    poses_group2[4, 1],
        target_group1[4, 1]], [
                                                                                    poses_group2[3, 1],
        target_group1[3, 1]], [
                                                                                    poses_group2[2, 1],
        target_group1[2, 1]], [
                                                                                    poses_group2[1, 1],
        target_group1[1, 1]], [
                                                                                    poses_group2[0, 1],
        target_group1[0, 1]]
    zs1, zs2, zs3, zs4, zs5, zs6, zs7, zs8, zs9, zs10, zs11, zs12, zs13, zs14 = [poses_group2[13, 2],
                                                                                 target_group1[13, 2]], [
                                                                                    poses_group2[12, 2],
        target_group1[12, 2]], [
                                                                                    poses_group2[11, 2],
        target_group1[11, 2]], [
                                                                                    poses_group2[10, 2],
        target_group1[10, 2]], [
                                                                                    poses_group2[9, 2],
        target_group1[9, 2]], [
                                                                                    poses_group2[8, 2],
        target_group1[8, 2]], [
                                                                                    poses_group2[7, 2],
        target_group1[7, 2]], [
                                                                                    poses_group2[6, 2],
        target_group1[6, 2]], [
                                                                                    poses_group2[5, 2],
        target_group1[5, 2]], [
                                                                                    poses_group2[4, 2],
        target_group1[4, 2]], [
                                                                                    poses_group2[3, 2],
        target_group1[3, 2]], [
                                                                                    poses_group2[2, 2],
        target_group1[2, 2]], [
                                                                                    poses_group2[1, 2],
        target_group1[1, 2]], [
                                                                                    poses_group2[0, 2],
        target_group1[0, 2]]
    # fig = plt.figure()
    # ax = fig.add_axes((0, 0, 1, 1), projection='3d')
    ax.plot(xs1, ys1, zs1, c='blue', marker='o',alpha=0.3)
    ax.plot(xs2, ys2, zs2, c='blue', marker='o',alpha=0.3)
    ax.plot(xs3, ys3, zs3, c='blue', marker='o',alpha=0.3)
    ax.plot(xs4, ys4, zs4, c='blue', marker='o',alpha=0.3)
    ax.plot(xs5, ys5, zs5, c='blue', marker='o',alpha=0.3)
    ax.plot(xs6, ys6, zs6, c='blue', marker='o',alpha=0.3)
    ax.plot(xs7, ys7, zs7, c='blue', marker='o',alpha=0.3)
    ax.plot(xs8, ys8, zs8, c='blue', marker='o',alpha=0.3)
    ax.plot(xs9, ys9, zs9, c='blue', marker='o',alpha=0.3)
    ax.plot(xs10, ys10, zs10, c='blue', marker='o',alpha=0.3)
    ax.plot(xs11, ys11, zs11, c='blue', marker='o',alpha=0.3)
    ax.plot(xs12, ys12, zs12, c='blue', marker='o',alpha=0.3)
    ax.plot(xs13, ys13, zs13, c='blue', marker='o',alpha=0.3)
    ax.plot(xs14, ys14, zs14, c='blue', marker='o',alpha=0.3)
    # plt.show()

def transform_pt(point, trans_mat):
    if(type(point) is not np.ndarray ):
        point=point.numpy()
    ap=[]
    for index in range(len(point)):
        a  = np.array([point[index][0],point[index][1],point[index][2],1])
        ap.append(np.dot(a, trans_mat)[:3])
    ap=torch.tensor(np.array(ap)).float()
    return ap

def double_view_sample_complex_mpjpe_analysis(poses_group1,poses_group2,target_group1,joint_id_index):
    poses_group1,poses_group2,target_group1=torch.from_numpy(np.array(poses_group1)),torch.from_numpy(np.array(poses_group2)),torch.from_numpy(np.array(target_group1))
    level1_distance=torch.sqrt(((poses_group1 - target_group1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    level2_distance =torch.sqrt(((poses_group2 - target_group1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    correct_level1,correct_level2=(poses_group1 - target_group1).mean(dim=0).cpu().numpy(),(poses_group2 - target_group1).mean(dim=0).cpu().numpy()
    return correct_level1,correct_level2

def double_view_sample_complex_statistical_analysis(poses_group1,poses_group2,target_group1,joint_id_index,axis_id):
    if(axis_id=='x'):
        axis_index=0
    elif(axis_id=='y'):
        axis_index = 1
    else:
        axis_index = 2
    level1_distance=poses_group1[:,joint_id_index,axis_index] - target_group1[:,joint_id_index,axis_index]
    level2_distance =poses_group2[:,joint_id_index,axis_index] - target_group1[:,joint_id_index,axis_index]
    level1_distance_sort,level2_distance_sort=np.sort(level1_distance),np.sort(level2_distance)
    correct_level1, correct_level2 = np.sum(level1_distance_sort) / len(level1_distance_sort), np.sum(level2_distance_sort) / len(level2_distance_sort)
    return correct_level1,correct_level2


def double_view_joint_MPI_Rotate_unify_coordinates(evaluation_accumulators_pred_j3d, evaluation_accumulators_target_j3d,evaluation_accumulators_camera_para, evaluation_accumulators_center,evaluation_accumulators_scaler):
    pred_j3ds_group1, target_j3ds_group1, camera_para_group1, box_center_group1, box_scaler_group1 = evaluation_accumulators_pred_j3d[1], evaluation_accumulators_target_j3d[1], evaluation_accumulators_camera_para[1], evaluation_accumulators_center[1], evaluation_accumulators_scaler[1]
    pred_j3ds_group2, target_j3ds_group2, camera_para_group2, box_center_group2, box_scaler_group2 = evaluation_accumulators_pred_j3d[2], evaluation_accumulators_target_j3d[2], evaluation_accumulators_camera_para[2], evaluation_accumulators_center[2], evaluation_accumulators_scaler[2]
    poses_group1,poses_group2,target_group1,boxes,cameras,box = [],[],[],[],[],{}
    for id_utem in range(len(target_j3ds_group1)):
        poses_group1.append(camera_to_world_frame(pred_j3ds_group1[id_utem], camera_para_group1[id_utem][0]['R'],camera_para_group1[id_utem][0]['T']))
        poses_group2.append(camera_to_world_frame(pred_j3ds_group2[id_utem], camera_para_group2[id_utem][0]['R'],camera_para_group2[id_utem][0]['T']))
        target_group1.append(camera_to_world_frame(target_j3ds_group1[id_utem], camera_para_group1[id_utem][0]['R'],camera_para_group1[id_utem][0]['T']))
        cameras.append(camera_para_group1[id_utem][0])
        box['scale'] = np.array(box_scaler_group1[id_utem])
        box['center'] = np.array(box_center_group1[id_utem])
        boxes.append(box)
        frame_item1,frame_item2,target_item1 =poses_group1[id_utem][[2,3,12]],poses_group2[id_utem][[2,3,12]],target_group1[id_utem][[2,3,12]]
        frame_item1_Q, frame_item2_Q,target_item1_Q=frame_item1[:2]-frame_item1[2],frame_item2[:2]-frame_item2[2],target_item1[:2]-target_item1[2]

        R = np.dot(np.linalg.inv(np.row_stack((frame_item1_Q, np.cross(*frame_item1_Q)))),np.row_stack((frame_item2_Q, np.cross(*frame_item2_Q))))
        t = frame_item2[2] - np.dot(frame_item1[2], R)
        M = np.column_stack((np.row_stack((R, t)),(0, 0, 0, 1)))
        poses_group1[id_utem]=transform_pt(poses_group1[id_utem],M).numpy()

        R = np.dot(np.linalg.inv(np.row_stack((target_item1_Q, np.cross(*target_item1_Q)))),np.row_stack((frame_item2_Q, np.cross(*frame_item2_Q))))
        t = frame_item2[2] - np.dot(target_item1[2], R)
        M = np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1)))
        target_group1[id_utem] = transform_pt(target_group1[id_utem], M).numpy()
        poses_group2[id_utem]=poses_group2[id_utem].numpy()
        double_view_sample_plot_inner(poses_group1[id_utem],poses_group2[id_utem],target_group1[id_utem])
        double_view_sample_plot_inner_distance(poses_group1[id_utem],poses_group2[id_utem],target_group1[id_utem])
    return np.array(poses_group1), np.array(poses_group2), np.array(target_group1)

def double_view_joint_h36m_Rotate_unify_coordinates (evaluation_accumulators_pred_j3d, evaluation_accumulators_target_j3d, evaluation_accumulators_center,evaluation_accumulators_scaler):
    pred_j3ds_group1, target_j3ds_group1, box_center_group1, box_scaler_group1 = evaluation_accumulators_pred_j3d[1], evaluation_accumulators_target_j3d[1], evaluation_accumulators_center[1], evaluation_accumulators_scaler[1]
    pred_j3ds_group2, target_j3ds_group2,  box_center_group2, box_scaler_group2 = evaluation_accumulators_pred_j3d[2], evaluation_accumulators_target_j3d[2], evaluation_accumulators_center[2], evaluation_accumulators_scaler[2]

    poses_group1, poses_group2, target_group1, boxes, cameras, box = [], [], [], [], [], {}
    for id_utem in range(len(target_j3ds_group1)):
        poses_group1.append(pred_j3ds_group1[id_utem])
        poses_group2.append(pred_j3ds_group2[id_utem])
        target_group1.append(target_j3ds_group1[id_utem])
        box['scale'] = np.array(box_scaler_group1[id_utem])
        box['center'] = np.array(box_center_group1[id_utem])
        boxes.append(box)
        frame_item1, frame_item2, target_item1 = poses_group1[id_utem][[2, 3, 12]], poses_group2[id_utem][[2, 3, 12]], target_group1[id_utem][[2, 3, 12]]
        frame_item1_Q, frame_item2_Q, target_item1_Q = frame_item1[:2] - frame_item1[2], frame_item2[:2] - frame_item2[2], target_item1[:2] - target_item1[2]

        R = np.dot(np.linalg.inv(np.row_stack((frame_item1_Q, np.cross(*frame_item1_Q)))),
                   np.row_stack((frame_item2_Q, np.cross(*frame_item2_Q))))
        t = frame_item2[2] - np.dot(frame_item1[2], R)
        M = np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1)))
        poses_group1[id_utem] = transform_pt(poses_group1[id_utem], M).numpy()

        R = np.dot(np.linalg.inv(np.row_stack((target_item1_Q, np.cross(*target_item1_Q)))),
                   np.row_stack((frame_item2_Q, np.cross(*frame_item2_Q))))
        t = frame_item2[2] - np.dot(target_item1[2], R)
        M = np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1)))
        target_group1[id_utem] = transform_pt(target_group1[id_utem], M).numpy()
    return np.array(poses_group1),np.array(poses_group2),np.array(target_group1)

def level1_2_correction_all_joint_xyz(level1,level2,target):
    correct_level1=np.loadtxt('C:\\Users\\Administrator\\Desktop\\论文备份\\高质量视图融合方法\\参数文件\\MPI_S1_correct_level1.txt')
    correct_level2 =np.loadtxt('C:\\Users\\Administrator\\Desktop\\论文备份\\高质量视图融合方法\\参数文件\\MPI_S1_correct_level2.txt')
    for index in range(len(level1)):
        level1[index]=level1[index]-correct_level1
        level2[index] = level2[index] - correct_level2

    return level1,level2

def level1_target_confidence_domain(level1,level2,target):
    for index in range(len(level1)):
        level1[index]-target[index]
    poses_group1, poses_group2, target_group1 = torch.from_numpy(level1), torch.from_numpy(level2), torch.from_numpy(target)
    return_level1_distance = (poses_group1 - target_group1).cpu().numpy()
    return_level2_distance = (poses_group2 - target_group1).cpu().numpy()
    return_inver_confidence_value1=(target_group1+target_group1-poses_group1).cpu().numpy()
    level1_distance = (poses_group1 - target_group1).cpu().numpy()
    level2_distance = (poses_group2 - target_group1).cpu().numpy()

    level1_distance_sort, level2_distance_sort = np.sort(level1_distance), np.sort(level2_distance)
    target_group1=target_group1.cpu().numpy()
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('X'),ax.set_ylabel('Y'),ax.set_zlabel('Z')

    x_axis,y_axis,z_axis=target_group1[0,:,0],target_group1[0,:,1],target_group1[0,:,2]
    # plt.plot(x_axis, y_axis, z_axis, c='green', marker='o')
    ax.scatter(x_axis, y_axis, z_axis, c='g' )
    plt.show()

    level1_plot, level2_plot = level1_distance_sort + target_group1[0, :, :], level2_distance_sort + target_group1[0, :,:]
    for index1 in range(0,len(level1_distance_sort),10):
        xs1,ys1,zs1=level1_plot[index1,:,0],level1_plot[index1,:,1],level1_plot[index1,:,2]
        ax.scatter(xs1, ys1, zs1, c='r', alpha=0.02)
    # plt.show()
    # plt.plot(xs2, ys2, zs2, c='red', marker='o')
    # ax.scatter(xs2, ys2, zs2, c='b', alpha=0.02)
    plt.show()
    # x1, y1 = range(len(level1_distance_sort)), level1_distance_sort.tolist()
    # x2, y2 = range(len(level2_distance_sort)), level2_distance_sort.tolist()

    return return_level1_distance,return_level2_distance,return_inver_confidence_value1
