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
from Utils.PreKeysNet_6pack_utils.pspnet import PSPNet
import torch.distributions as tdist
import copy

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):  # 图片提取特征

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):  # 图片和点云一起提取特征
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous() #128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous()  # 128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


class KeyNet(nn.Module):
    def __init__(self, num_points, num_key, num_cates):
        super(KeyNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        self.num_cates = num_cates

        self.sm = torch.nn.Softmax(dim=1)

        self.kp_1 = torch.nn.Conv1d(160, 90, 1)
        self.kp_2 = torch.nn.Conv1d(90, 3 * num_key, 1)

        self.att_1 = torch.nn.Conv1d(160, 90, 1)
        self.att_2 = torch.nn.Conv1d(90, 1, 1)

        self.sm2 = torch.nn.Softmax(dim=1)

        self.num_key = num_key

        self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).view(1, 1, 3).repeat(
            1, self.num_points, 1)

    def forward(self, img, choose, x):
        # img 是roi-img 裁剪的
        # choose 是裁剪图像中的物体
        # x 是观测的物体点云

        # num_anc = len(anchor[0])
        out_img = self.cnn(img)  # 图片特征提取的网络

        bs, di, _, _ = out_img.size()
        # 这个spd也有，是能把图片中的实例特征从整个图片中提取出来吗？神奇，不过缺了进一步的特征提取
        emb = out_img.view(bs, di, -1)
        # choose = torch.tensor(choose, dtype=torch.int64)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        # 这个应该就是点云特征的提取
        # 先将点云中心化
        aveCentor = torch.mean(x, dim=1).unsqueeze(1)
        x = x - aveCentor
        # 提取特征
        x = x.transpose(2, 1).contiguous()
        feat_x = self.feat(x, emb)  # 点云和图片一块提取特征了（denseFusion？ ）
        feat_x = feat_x.transpose(2, 1).contiguous()  # 24, 1028, 160


        # 注意力机制，对每个点
        loc = x.transpose(2, 1).contiguous()
        weight = self.sm(-1.0 * torch.norm(loc, dim=2)).contiguous()

        feat_x = torch.sum((feat_x * weight.unsqueeze(-1)), dim=1).contiguous()  # 给每个点对应的特征权重 24, 160

        # 通过kp_1 和kp_2 找点 因为和kp_2 的输出就是 点个数*3
        feat_x = feat_x.unsqueeze(-1)
        kp_feat = F.leaky_relu(self.kp_1(feat_x))
        kp_feat = self.kp_2(kp_feat)
        kp_feat = kp_feat.transpose(2, 1).contiguous()
        kp_x = kp_feat.view(bs, -1, 3).contiguous()  # bs，8 ,3

        # 将关键点去中心化
        kp_x = kp_x + aveCentor

        return kp_x  # 第一个参数 ：预测的关键点。 第二个参数： 预测的质心，是为了质心loss，处理跟踪偏移的。 最后一个参数 ： 大概是为了给质心加权重吧