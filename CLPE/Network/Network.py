import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
import pytorch3d.loss

# from einops import rearrange

# from Utils.PreKeysNet_6pack import KeyNet
from Utils.PreKeysNet_GPV import KeyNet
from Utils.CatIcpNet_GPV import CatIcpNet
from Utils.SPDResNorNet_RBPose import SPD_BboxNormal_Net
from Utils.LossCatSim_KeyPtsDefor import CatSimLoss
from Utils.LossRts_GPV import PoseSizeLoss
from Utils.SPDResNorLoss_RBPose import BboxGeometryLoss  # 用包围盒法向等约束增强旋转平移约束
from Utils.utils import InvTransform9D, SavePoints_asPLY, get_gt_v


# from ..utils.utils import normalize_to_box, sample_farthest_points

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class KeysNetwork(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--lambda_init_points", type=float, help="", default=2.0)
        parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
        parser.add_argument("--lambda_influence_predict_l2", type=float, help="", default=1e6)
        parser.add_argument("--iterations_init_points", type=float, help="", default=200)
        parser.add_argument("--no_optimize_cage", action="store_true", help="")
        parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        parser.add_argument("--n_fps", type=int, help="")
        return parser

    def __init__(self, FLAGS):  # 网络和优化器的初始化
        super(KeysNetwork, self).__init__()
        self.FLAGS = FLAGS
        self.init_networks(FLAGS.num_points, FLAGS.num_kp, FLAGS.num_cates)
        self.init_loss()
        self.init_optimizer()

    def init_networks(self, num_points, num_key, num_cates):  # 初始化网络
        # 使用图片和点云获得关键点的网络
        # self.keyNet = KeyNet(num_points, num_key, num_cates)  # 6pack的方式RGBD的
        self.keyNet = KeyNet()  # 6pack的方式RGBD的
        # 类ICP网络
        self.icpNet = CatIcpNet()
        # 变到实例模型和预测bbox法向的网络
        self.deformNet = SPD_BboxNormal_Net()

    def init_loss(self):
        # 变形相似度loss，获取观测不变性的关键点， deformed变形模型是不是一样可以？
        self.catSimLoss = CatSimLoss()
        self.l1loss = nn.L1Loss()  # 与gt实例关键点做loss
        self.poseSizeLoss = PoseSizeLoss()
        # 变形相关的loss和bbox法向相关的loss
        self.bboxGeometryLoss = BboxGeometryLoss()


        # pose和size的loss
    def init_optimizer(self):  # 初始化优化器
        # 关键点优化器
        params = [{"params": self.keyNet.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.FLAGS.lr)

        # icp优化器
        icpNetparams = [{"params": self.icpNet.parameters()}]
        self.icpNet_optimizer = torch.optim.Adam(icpNetparams, lr=self.FLAGS.lr)
        self.icpNet_optimizer.add_param_group({'params': self.deformNet.parameters(), 'lr': self.FLAGS.lr})


    # 网络的前向
    def forward(self, img, choose, pts, catKeys,obj_id,
                prior=None, mean_shape=None, trainFlag=1):    # 这一行的参数是变形相关网络额外的信息
        self.obj_id = obj_id
        # 用网络获得实例关键点
        # keyPoints = self.keyNet(img, choose, pts)  # 6pack的方式RGBD的
        keyPoints = self.keyNet(pts, obj_id)
        self.keyPoints = keyPoints

        # 类关键点和实例关键点进行类ICP网络获得位姿
        # 因为用CateSimLoss预测的实例关键点实际使笼形关键点，所以要把类关键点变为类笼形关键点  [事情好像没那么简单，先不弄，话说笼形变形的约束可以直接变成模型变形的形变把，再结合最近的文章添加关键点分布的约束]
        self.p_green_R, self.p_red_R, self.f_green_R, self.f_red_R, self.Pred_T, self.Pred_s, keysFeats = self.icpNet(catKeys, keyPoints.detach())

        if trainFlag == 1:  # 只有在训练的时候，训练这个网络，测试不用
            # 用预测的关键点，预测实例模型，对应关系，以及bbox法向的残差
            self.deformNetOutput = self.deformNet(keyPoints, obj_id, prior, mean_shape, keysFeats, self.p_green_R, self.p_red_R, self.f_green_R, self.f_red_R)

        outputs = {
            'preInsKeys': keyPoints,  # 预测的关键点
            'p_green_R': self.p_green_R,  # 旋转轴1
            'p_red_R': self.p_red_R,  # 旋转轴2
            'Pred_T': self.Pred_T,  # 位移
            'Pred_s': self.Pred_s,  # 尺寸  # 这个尺寸是残差
            'f_green_R': self.f_green_R,  # 旋转轴1的置信度
            'f_red_R': self.f_red_R  # 旋转轴2的置信度
        }

        return outputs

    def compute_loss(self, gts, gtR, gtt, catModel, catKeys, insKeys, catInfluence, insModel, sym, gt_lwh):  # 计算各自的loss，存为字典
        '''
        :param gts:  真实尺寸， 指长方体对角线
        :param gtR:  gt旋转
        :param gtt:  gt平移
        :param catModel:  类模型
        :param catKeys:  类关键点
        :param insKeys:  gt实例关键点
        :param catInfluence: 用于类模型变形的影响因子
        :param insModel:  gt实例模型
        :param sym:  类对称性
        :param gt_lwh:  gt尺寸 指长宽高
        :return:
        '''

        losses = {}

        ############### 关键点的loss ： 使用模型相似度做监督###############
        if self.FLAGS.simCatLossFlag > 0:
            # 将tensor类型的Rt合成T
            bs = gtR.shape[0]
            gtRt = torch.zeros([bs, 3, 4], device=gtR.device)  #  新建到相同的device上
            gtRt[:, :3, :3] = gtR
            gtRt[:, :3, 3] = gtt

            # 将实例点云通过gt Rts变换到NOCS中
            insKeysinCam = self.keyPoints
            preKeyinNOCS = InvTransform9D(insKeysinCam, gts, gtRt)
            # 计算变形的相似度loss
            outputs = self.catSimLoss(catModel, catKeys, catInfluence, preKeyinNOCS, insModel)
            catSimLoss = outputs["simLoss"]
            losses['catSimLoss'] = self.FLAGS.catSimLossWeight * catSimLoss  # 第一个参数是权重

        ############### 关键点的loss ： L1loss###############
        if self.FLAGS.L1KeysLossFlag > 0:
            # # 保存gt实例关键点，确定关键点的正确性
            # print(type(insKeys), insKeys.shape)
            # print(type(insKeys.cpu().numpy()[0]), insKeys.cpu().numpy()[0].shape, insKeys.cpu().numpy()[0])
            # SavePoints_asPLY(catModel.cpu().numpy()[0], '/data1/jl/6pack/SelfResults/others/ProveinTrain/catModel.ply')
            # SavePoints_asPLY(catKeys.cpu().numpy()[0], '/data1/jl/6pack/SelfResults/others/ProveinTrain/catKeys.ply')
            # SavePoints_asPLY(insKeys.cpu().numpy()[0], '/data1/jl/6pack/SelfResults/others/ProveinTrain/gtInsKeys1.ply')
            # SavePoints_asPLY(insKeys.cpu().numpy()[1], '/data1/jl/6pack/SelfResults/others/ProveinTrain/gtInsKeys2.ply')
            # SavePoints_asPLY(insKeys.cpu().numpy()[2], '/data1/jl/6pack/SelfResults/others/ProveinTrain/gtInsKeys3.ply')
            # SavePoints_asPLY(insKeys.cpu().numpy()[3], '/data1/jl/6pack/SelfResults/others/ProveinTrain/gtInsKeys4.ply')
            # asdf

            # 将tensor类型的Rt合成T
            bs = gtR.shape[0]
            gtRt = torch.zeros([bs, 3, 4], device=gtR.device)  # 新建到相同的device上
            gtRt[:, :3, :3] = gtR
            gtRt[:, :3, 3] = gtt
            # 将实例点云通过gt Rts变换到NOCS中
            insKeysinCam = self.keyPoints
            preKeyinNOCS = InvTransform9D(insKeysinCam, gts, gtRt)
            # 预测的关键点inNOCS:preKeyinNOCS， gt的实例关键点
            l1KeysLoss = self.l1loss(preKeyinNOCS, insKeys)
            # initKeysLoss = pytorch3d.loss.chamfer_distance(preKeyinNOCS, insKeys)[0]
            losses['l1KeysLoss'] = self.FLAGS.l1KeysLossWeight * l1KeysLoss  # 第一个参数是权重

            # # 预测的关键点inNOCS:preKeyinNOCS， gt的实例关键点
            # initKeysLoss = pytorch3d.loss.chamfer_distance(preKeyinNOCS, insKeys)[0]
            # losses['initKeysLoss'] = self.FLAGS.initKeysLossWeight * initKeysLoss  # 第一个参数是权重

        ############### 位姿的loss ： 旋转、平移和尺寸的loss约束###############
        if self.FLAGS.sRtLossFlag > 0:
        # 旋转、平移和尺寸的loss约束
            pred_list = {
                'Rot1': self.p_green_R,
                'Rot1_f': self.f_green_R,
                'Rot2': self.p_red_R,
                'Rot2_f': self.f_red_R,
                'Tran': self.Pred_T,
                'Size': self.Pred_s,  # 这个size是长宽高
            }
            gt_green_v, gt_red_v = get_gt_v(gtR)

            gt_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Tran': gtt,
                'Size': gt_lwh,  # 这个地方是长宽高的size  !!这个地方给错了！应该给包围盒的残差！！！！！！！
            }
            pSLoss = self.poseSizeLoss(pred_list, gt_list, sym)
            # losses['poseSiseLoss'] = pSLoss
            losses['poseSiseLoss'] = pSLoss["Rot1"] +pSLoss["Rot1_cos"] +pSLoss["Rot2"] +pSLoss["Rot2_cos"] + \
                                     pSLoss["Rot_r_a"] +pSLoss["Tran"] +pSLoss["Size"] +pSLoss["R_con"]
            # 将pSLoss中的小loss放进来，好写进tnesorboard中
            for k, v in pSLoss.items():
                losses[k] = v

        ############### 位姿的loss ： 位姿增强、Bbox的法向约束###############
        if self.FLAGS.bboxNorLossFlag > 0:
            # 给到网络
            pred_list = {
                # shapeCorLoss 需要的数据
                'assign_mat': self.deformNetOutput['assign_mat'],  # [bs, 8, 1024]
                'deltas': self.deformNetOutput['deform_field'],  # [bs, 1024, 3]

                # enhancePoseLoss 需要的数据
                'Rot1': self.p_green_R,
                'Rot1_f': self.f_green_R.detach(),
                'Rot2': self.p_red_R,
                'Rot2_f': self.f_red_R.detach(),
                'Tran': self.Pred_T,
                'Recon': self.deformNetOutput['recon'],

                # resBboxNorLoss 需要的数据
                'face_shift': self.deformNetOutput['face_shift'],
                'face_shift_delta': self.deformNetOutput['face_shift_delta'],
                'F_log_var': self.deformNetOutput['face_log_var'],

            }
            gt_list = {
                # shapeCorLoss 需要的数据
                'prior': catModel,  # 先验模型
                'nocs': insKeys,  # 关键点在NOCS下的3D坐标
                'model': insModel,  # gt的实例模型

                # enhancePoseLoss 需要的数据D
                'Points': self.keyPoints,  # 预测的关键点inCam
                'R': gtR,
                'T': gtt,

                # resBboxNorLoss 需要的数据
                'Size': gt_lwh  # 这个是gt尺寸的长宽高
            }

            # 不做区分，都算，优化的时候不加就是了
            shapeCorLoss, enhancePoseLoss, resBboxNorLoss = self.bboxGeometryLoss(sym, pred_list, gt_list,
                                                                                      self.obj_id, resBboxNorFlag=1)

            # 将Loss中的小loss放进来，好写进tnesorboard中
            Losstmp = 0
            for k, v in shapeCorLoss.items():
                losses[k] = v
                Losstmp = Losstmp + v
            losses['shapeCorLoss'] = Losstmp

            Losstmp = 0
            for k, v in enhancePoseLoss.items():
                Losstmp = Losstmp + v
            losses['enhancePoseLoss'] = Losstmp

            Losstmp = 0
            for k, v in resBboxNorLoss.items():
                Losstmp = Losstmp + v
            losses['resBboxNorLoss'] = Losstmp

        return losses

    def _sum_losses(self, losses, names):  # loss求和，用以反向传播
        return sum(v for k, v in losses.items() if k in names)

    def optimize(self, losses, iteration):  # 每步的优化，反向传播

        # 先优化关键点的网络
        if iteration < self.FLAGS.iterations_init_points:
            self.optimizer.zero_grad()
            loss = self._sum_losses(losses, ['l1KeysLoss'])
            loss.backward()
            self.optimizer.step()

        # 再优化位姿
        if iteration >= self.FLAGS.iterations_init_points:
            self.icpNet_optimizer.zero_grad()
            if iteration <= self.FLAGS.iterations_fineTuningPose:
                loss = self._sum_losses(losses, ['shapeCorLoss', 'poseSiseLoss', 'enhancePoseLoss'])
            else:
                loss = self._sum_losses(losses, ['shapeCorLoss', 'poseSiseLoss', 'enhancePoseLoss', 'resBboxNorLoss'])
            loss.backward()
            # self.optimizer.step()
            self.icpNet_optimizer.step()

    # # 保存网络的模型
    # def NetSave(self, path, epoch):
    #     torch.save(self.keyNet.state_dict(), '{0}/model_{1:02d}.pth'.format(path, epoch))
    #     # FLAGS.model_save
    # 恢复部分网络预训练模型
    # state_dict = torch.load(FLAGS.resumePart_model) 要先读入pth中的参数
    def NetResume(self, state_dict):
        self.keyNet.load_state_dict(state_dict)