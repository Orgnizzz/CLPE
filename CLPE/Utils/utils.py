import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import os
from collections import OrderedDict

# 这个应该是，根据裁剪的mask 对裁剪的深度图，进行采样，输出物体的点云pc，和采样的序列choose
def PC_sample(obj_mask, Depth, camK, coor2d, rand_num=1028, sample_method='basic'):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''

    Depth = Depth.unsqueeze(0)
    obj_mask = obj_mask.unsqueeze(0)
    camK = camK.unsqueeze(0)
    coor2d = coor2d.unsqueeze(0)

    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    '''
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    samplenum = rand_num

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)
    chooses = torch.zeros([bs, samplenum], dtype=torch.int64, device=Depth.device)
    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        # basic sampling
        if sample_method == 'basic':
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            p_select = p_n_now[choose, :]
        else:
            p_select = None
            raise NotImplementedError

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]

        PC[i, ...] = p_select[:, :3]
        chooses[i, ...] = torch.tensor(choose, dtype=torch.float32)

    PC = PC.squeeze(0)
    chooses = chooses.squeeze(0)

    return PC / 1000.0, chooses


#保存点云为ply文件
def SavePoints_asPLY(points, path, normals=None):
    # points ： np类型点云n*3
    # path ： 保存的路径，如： "/home/zzh/awsl-JL/object-deformnet-master/object-deformnet-master/data/results/"+"/points.ply"
    # normals : 点云的法向、单位向量 np类型点云n*3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        if normals == None:
            print('no Normal')
    except:
        if points.shape[0] == normals.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals)
            print('existing Normal')
        else:
            print('error: num of points and normal is not equal')

    o3d.io.write_point_cloud(path, pcd) #保存点云

# 函数 ： 将tensor类型的点云，经过尺寸，旋转和平移，变换
# 输入 ：tensor点云：batch*num*3；尺度 ： batch ； 位姿 ： batch*3*4
def transform9D(trPoints, trSize, trPose):
    # 先将关键点标签变换到相同size和位姿下
    trSizeTmp = trSize.unsqueeze(-1).unsqueeze(-1)
    keyPtsinsize = trPoints * trSizeTmp
    now_keyPts = torch.bmm(trPose[:, :3, :3], keyPtsinsize.transpose(1, 2)) + trPose[:, :3, 3].unsqueeze(-1)
    now_keyPts = now_keyPts.transpose(1, 2)
    return now_keyPts
# 说明：A坐标系到B坐标系的9D位姿变换为sRt， 将B坐标系的点云变到A坐标系，通过sRt*pts 是不对的！
# 输入 ： tensor点云：batch*num*3；尺度 ： batch ； 位姿 ： batch*3*4
def InvTransform9D(trPoints, trSize, trPose):  # 输入 ： 现在的点云。 之前到现在的尺寸和位姿
    # 先将关键点标签变换到相同size和位姿下
    trSizeTmp = trSize.unsqueeze(-1).unsqueeze(-1)
    keyPtsinstmp = trPoints - trPose[:, :3, 3].unsqueeze(-1).transpose(1, 2)  # 先减去位移
    keyPtsinsize = keyPtsinstmp * (1/trSizeTmp)  #进行比例的变换
    now_keyPts = torch.bmm(trPose[:, :3, :3].transpose(1, 2), keyPtsinsize.transpose(1, 2))  # 乘旋转的转置
    now_keyPts = now_keyPts.transpose(1, 2)
    return now_keyPts



# 取出旋转矩阵的两个方向向量
# 输入 ： Rs ： bs x 3 x 3 旋转， tensor
def get_gt_v(Rs, axis=2):
    bs = Rs.shape[0]  # bs x 3 x 3
    # TODO use 3 axis, the order remains: do we need to change order?
    if axis == 3:
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    else:
        assert axis == 2
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    gt_green = gt_vec[:, 3:6]
    gt_red = gt_vec[:, (6, 7, 8)]
    return gt_green, gt_red


# 网络有很多网络组成，保存预训练模型
def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    例子 ： save_network(net, checkpoints_dir, network_label="net", epoch_label="final")
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included

    From https://github.com/yifita/deep_cage
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()

# 网络由很多网络组陈，恢复预训练模型
def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step

    From https://github.com/yifita/deep_cage
    """
    # warnings.DeprecationWarning("load_network is deprecated. Use module.load_state_dict(strict=False) instead.")
    if isinstance(path, str):  # isinstance判断对象是否是已知类型
        # logger.info("loading network from {}".format(path))
        if path[-3:] == "pth":
            loaded_state = torch.load(path)
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
        else:
            loaded_state = np.load(path).item()
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if len(missingkeys)>0:
        logger.warn("load_network {} missing keys".format(len(missingkeys)), "\n".join(missingkeys))
    if len(unexpectedkeys)>0:
        logger.warn("load_network {} unexpected keys".format(len(unexpectedkeys)), "\n".join(unexpectedkeys))