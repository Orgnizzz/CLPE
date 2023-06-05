import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from Utils.SPDResNorNet_RBPose_Utils.FaceRecon import FaceRecon
from Utils.SPDResNorNet_RBPose_Utils.pcl_encoder import PCL_Encoder
from Utils.SPDResNorNet_RBPose_Utils.shape_prior_utils import get_nocs_from_deform, get_face_dis_from_nocs, get_face_shift_from_dis

# 提取先验模型特征的网络
class PriorExtraNet(nn.Module):
    def __init__(self):
        super(PriorExtraNet, self).__init__()
        self.pcl_encoder_obs = PCL_Encoder()  # 提取先验模型和观测的特征

    def forward(self, obj_id, prior):
        '''
        obj_id：类别。
        prior：先验模型
        localFeat, globalFeat: 模型的局部特征、模型的全局特征
        '''
        localFeat, globalFeat = self.pcl_encoder_obs(prior, obj_id)
        return localFeat, globalFeat

# 控制形变和对应关系的网络
class Deform_CorrespondingNet(nn.Module):
    '''
    prior_num: 先验模型的点的个数
    '''
    def __init__(self, n_cat=6, prior_num=1024):
        super(Deform_CorrespondingNet, self).__init__()
        self.n_cat = n_cat

        self.assignment = nn.Sequential(  # 找对应矩阵的网络
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.n_cat * prior_num, 1),
        )
        self.deformation = nn.Sequential(  # 变形的网络
            nn.Conv1d(3334, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.n_cat * 3, 1),
        )

    def forward(self, obj_id, prior, viewGlobalFeats, priorLocalFeats, priorGlobalFeats):
        '''obj_id, prior : 类别，先验模型
        viewGlobalFeats, ： 观测的局部特征和全局特征  实例和类关键点的特征：bs,8,1286
        priorLocalFeats, priorGlobalFeats ： 先验模型的局部特征和全局特征
        nocs_pred : 观测点云在nocs空间中对应点
        '''
        bs, p_num = prior.shape[0], 8  # 这个8是关键点的个数
        prior_num = prior.shape[1]
        feat_global_obs = viewGlobalFeats
        feat_prior, feat_global_prior = priorLocalFeats, priorGlobalFeats

        feat_obs_fuse = feat_global_obs  # bs,8,1286
        # prior feature??
        # feat_prior : [bs, 1024, 1286] feat_global_prior : [bs, 512]
        feat_prior_fuse = torch.cat((feat_prior, feat_global_prior.repeat(prior_num, 1, 1).permute(1, 0, 2)),
                                    dim=2)  # bs, 1024, 1798

        # assign matrix
        assign_feat = torch.cat((feat_obs_fuse, feat_global_prior.repeat(p_num, 1, 1).permute(1, 0, 2)),
                                dim=2).permute(0, 2, 1)  # bs x 2048 x 8

        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, prior_num, p_num).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts

        # index = obj_id.long() + torch.arange(bs, dtype=torch.long) * self.n_cat  # 在cpu上用这个，gpu上用下面这个
        index = obj_id.long() + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv    bs, 8, 1024

        # deformation field
        # feat_prior_fuse : 2, 1024, 1798  feat_global_obs : 2, 8, 1536
        deform_feat = torch.cat((feat_prior_fuse, feat_global_obs.repeat(1, 128, 1)),
                                dim=2).permute(0, 2, 1)  # bs x 2342 x n_pts  bs, 3334, 1024
        deform_field = self.deformation(deform_feat)
        deform_field = deform_field.view(-1, 3, prior_num).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv
        deform_field = torch.index_select(deform_field, 0, index)  # bs x 3 x nv
        deform_field = deform_field.permute(0, 2, 1).contiguous()  # bs x nv x 3
        nocs_pred = get_nocs_from_deform(prior, deform_field, assign_mat)  # bs, 8, 3

        return nocs_pred, assign_mat, deform_field

# 预测包围盒的法向量残差
class ResNormalNet(nn.Module):
    def __init__(self):
        super(ResNormalNet, self).__init__()
        self.face_recon = FaceRecon()  # 预测表面法向

    def forward(self, nocs_pred, p_green_R, p_red_R, f_green_R, f_red_R, viewGlobalFeats, points, mean_shape):
        '''
        nocs_pred : 观测点云在nocs空间中对应点
        , p_green_R, p_red_R, f_green_R, f_red_R, : 两个轴预测的旋转和对应的置信
        viewLocalFeats, viewGlobalFeats ： 观测相关的局部特征和全局特征
        points ： 观测点云,
        PC_seman ： 【这个应该是采样出来的一些点融合到特征里面以提高精度的，或可先去掉】
        mean_shape ： 应该是平均模型的长宽高
        '''
        # 参数
        feat_global_obs = viewGlobalFeats
        # 网络处理
        face_dis_prior = get_face_dis_from_nocs(nocs_pred, mean_shape)

        face_shift_prior = get_face_shift_from_dis(face_dis_prior, p_green_R, p_red_R, f_green_R, f_red_R,
                                                   use_rectify_normal=0)  # face_shift_prior : bs, 8, 18

        recon, face_shift_delta, face_log_var, mask = self.face_recon(feat_global_obs,
                                                                      feat_global_obs,
                                                                      points - points.mean(dim=1, keepdim=True),
                                                                      face_shift_prior)
        recon = recon + points.mean(dim=1, keepdim=True)

        # handle face
        face_shift = face_shift_delta + face_shift_prior  # 最终预测的指向包围盒的方向

        return face_shift, recon, face_shift_delta, face_log_var



# 组成一个网络： 提取先验模型的特征； 加上观测相关特征进行重建和对应； 根据NOCS中的观测预测BBOX法向残差
class SPD_BboxNormal_Net(nn.Module):
    def __init__(self, n_cat=6, prior_num=1024):
        super(SPD_BboxNormal_Net, self).__init__()
        self.priorExtraNet = PriorExtraNet()  # 提取先验模型特征
        self.deform_CorrespondingNet = Deform_CorrespondingNet(n_cat, prior_num)  # 变形和找对应关系
        self.resNormalNet = ResNormalNet()  # 预测Bbox的法向

    def forward(self, points, obj_id, prior, mean_shape, viewGlobalFeats, p_green_R, p_red_R, f_green_R, f_red_R):
        '''
        points, obj_id, prior, mean_shape, : 观测点云(或预测的关键点？)、类别、先验模型、平均尺寸
        viewLocalFeats, viewGlobalFeats, ： 观测的局部特征和全局特征
        p_green_R, p_red_R, f_green_R, f_red_R ： 预测的两个轴和置信
        face_shift ： Bbox的法向
        '''
        priorLocalFeats, priorGlobalFeats = self.priorExtraNet(obj_id, prior)
        nocs_pred, assign_mat, deform_field = self.deform_CorrespondingNet(obj_id, prior, viewGlobalFeats, priorLocalFeats, priorGlobalFeats)
        face_shift, recon, face_shift_delta, face_log_var = self.resNormalNet(nocs_pred, p_green_R, p_red_R, f_green_R, f_red_R, viewGlobalFeats, points, mean_shape)

        output = {
            'face_shift': face_shift,
            'assign_mat': assign_mat,
            'deform_field': deform_field,  # 先验模型到实例模型的残差
            'recon': recon,  # 使用特征重建得到的关键点
            'face_shift_delta': face_shift_delta,
            'face_log_var': face_log_var,

        }

        return output



if __name__ == "__main__":
    # 初始化网络
    testNet = SPD_BboxNormal_Net()
    print('init Net ok')
    # 生成输入
    points = torch.rand(2, 8, 3)  # 实例关键点
    obj_id = torch.Tensor([3, 4])   # 类别
    # print(obj_id.shape)
    # asdf
    prior = torch.rand(2, 1024, 3)  # 先验模型
    mean_shape = torch.rand(2, 3)  # 模型的平均尺寸
    viewGlobalFeats = torch.rand(2, 8, 1536)  # 这个打算给pre实例和gt类关键点拼接的特征
    p_green_R = torch.rand(2, 3)
    p_red_R = torch.rand(2, 3)
    f_green_R = torch.rand(2)
    f_red_R = torch.rand(2)

    # 输入到网络
    face_shift = testNet(points, obj_id, prior, mean_shape, viewGlobalFeats, p_green_R, p_red_R, f_green_R, f_red_R)
    print(face_shift.shape)
    print('finally dim is ok')






