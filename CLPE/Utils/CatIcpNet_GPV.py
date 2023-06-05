import os
import _pickle as cPickle
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import math
import torch
import open3d as o3d
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# from CatIcpNet_GPV_utils.FSNetArchs import KeysFeatsExtra, EstiSt, RotGreen, RotRed
# from CatIcpNet_GPV_utils.PoseR import Rot_red, Rot_green
# from CatIcpNet_GPV_utils.PoseTs import Pose_Ts
from Utils.CatIcpNet_GPV_utils.FSNetArchs import KeysFeatsExtra, EstiSt, RotGreen, RotRed
from Utils.CatIcpNet_GPV_utils.PoseR import Rot_red, Rot_green
from Utils.CatIcpNet_GPV_utils.PoseTs import Pose_Ts



# 将类ICP封装起来, FSNet的
class CatIcpNet(nn.Module):  # 图片提取特征

    def __init__(self):
        super(CatIcpNet, self).__init__()
        # 网络初始化
        self.keyFeatsNet = KeysFeatsExtra()  # 特征提取
        # 旋转、平移、尺寸的网络
        self.rot_green = Rot_green()  # 旋转轴1的
        self.rot_red = Rot_red()  # 旋转轴2的
        self.ts = Pose_Ts()  # 位移和尺度的

    def forward(self, sourceKey, targetKey):
        '''
         sourceKey targetKey : tensor : batch * 8 * 3
         sourceKey ： 类关键点； targetKey： 实例关键点
         注意 ： 训练的时候batch不能为1！因为有batchNormal！！
        '''

        # 使用3DGCN提取特征
        sourceFeats = self.keyFeatsNet(sourceKey)  # 64, 8, 768
        targetFeats = self.keyFeatsNet(targetKey)
        keysFeats = torch.cat([sourceFeats, targetFeats], dim=2)  # 64, 8, 1536

        #  旋转预测
        green_R_vec = self.rot_green(keysFeats.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(keysFeats.permute(0, 2, 1))  # b x 4

        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = torch.sigmoid(green_R_vec[:, 0])  # 小网络预测置信度
        f_red_R = torch.sigmoid(red_R_vec[:, 0])

        # 平移和尺寸
        feat_for_ts = torch.cat([keysFeats, targetKey - targetKey.mean(dim=1, keepdim=True),
                                 sourceKey - sourceKey.mean(dim=1, keepdim=True)], dim=2)
        T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        Pred_T = T + targetKey.mean(dim=1)  # bs x 3

        Pred_s = s  # this s is not the object size, it is the residual
        return p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s, keysFeats
        # p_green_R, p_red_R: bs, 3
        # f_green_R, f_red_R : bs
        # Pred_T : bs, 3
        # Pred_s : bs, 3




if __name__ == "__main__":
    # 生成仿真数据
    cateKey = torch.rand([2, 8, 3]).cuda()
    insKey = torch.rand([2, 8, 3]).cuda()
    # 初始化网络
    testNet = CatIcpNet().cuda()
    testNet.train()
    # 前向
    p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s = testNet(cateKey, insKey)
    print(p_green_R.shape, p_red_R.shape, f_green_R.shape, f_red_R.shape, Pred_T.shape, Pred_s.shape)
    asdf
    #   预测的维度
    # torch.Size([2, 3]) torch.Size([2, 3]) torch.Size([2]) torch.Size([2]) torch.Size([2, 3]) torch.Size([2, 3])
















# #函数说明：生成随机的R，t， 输出为np类型的4*4的T矩阵
# def RandRT():
#     #输出T ： np类型的4*4的T矩阵
#     r = [(random.random()-0.5)*100, (random.random()-0.5)*100 ,(random.random()-0.5)*100]
#     t = [(random.random()-0.5)*0.6, (random.random()-0.5)*0.6,(random.random()-0.5)*0.6] #给平均75度。50cm的扰动
# #         r = [0, 0 ,0]
# #         t = [0, 0 ,0]#[(random.random()-0.5)*0.1, (random.random()-0.5)*0.1,(random.random()-0.5)*0.1]
#     t = np.array(t)
#     r4 = Euler2Rotation(r)
#     T = np.array([[r4[0,0],r4[0,1],r4[0,2],t[0]], [r4[1,0],r4[1,1],r4[1,2],t[1]], [r4[2,0],r4[2,1],r4[2,2],t[2]], [0, 0, 0, 1]])
# #         s1 = np.cbrt(np.linalg.det(T[:3, :3])) #把旋转变成正交矩阵
# #         T[:3, :3] = T[:3, :3]/s1
#     return T
# #欧拉角转旋转矩阵 https://blog.csdn.net/shyjhyp11/article/details/111701127都有
# #from scipy.spatial.transform import Rotation as R
# def Euler2Rotation(r):
#     #r, list 1*3,度
#     r4 = R.from_euler('zxy', r, degrees=True)
#     rm = r4.as_matrix()
#     return rm
#
# #svd计算两点云的刚体变换，两点云点要是一一对应的
# def compute_rigid_transform(source_pc, target_pc):
# #source_pc，target_pc： 1024*3， numpy类型
#     if source_pc.shape[0] != 3:
#         source_pc = np.transpose(source_pc)
#         target_pc = np.transpose(target_pc)
#
#     T = np.eye(4)
#     R, t = rigid_transform_3D(source_pc, target_pc)
#     T[0:3, 0:3], T[0:3, -1] = R, np.squeeze(t)
#
#     return T
# def rigid_transform_3D(A, B):
#     assert A.shape == B.shape
#
#     num_rows, num_cols = A.shape
#     if num_rows != 3:
#         raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
#
#     num_rows, num_cols = B.shape
#     if num_rows != 3:
#         raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
#
#     # find mean column wise
#     centroid_A = np.mean(A, axis=1)
#     centroid_B = np.mean(B, axis=1)
#
#     # ensure centroids are 3x1
#     centroid_A = centroid_A.reshape(-1, 1)
#     centroid_B = centroid_B.reshape(-1, 1)
#
#     # subtract mean
#     Am = A - centroid_A
#     Bm = B - centroid_B
#
#     H = Am @ np.transpose(Bm)
#
#     # sanity check
#     #if linalg.matrix_rank(H) < 3:
#     #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))
#
#     # find rotation
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
#
#     # special reflection case
#     if np.linalg.det(R) < 0:
#         print("det(R) < R, reflection detected!, correcting for it ...")
#         Vt[2,:] *= -1
#         R = Vt.T @ U.T
#
#     t = -R @ centroid_A + centroid_B
#
#     return R, t
#
# #备份
# # noisePoints = noiseT[:3, :3].dot(instanceKey[inst].T) + noiseT[:3, 3, np.newaxis]     # 3*8的np
#
# # @#计算两np类型的位姿矩阵的旋转平移误差
# def ErrNcmNdeg(preTrans, gtTrans):
#     # 输入 ： preTrans ： 预测的， 3*4的numpy， 前3*3是旋转， 后3*1是平移， 矩阵要正交！不正交怎么也算不对！
#     #        gtTrans :  gt ， 3*4的numpy， 前3*3是旋转， 后3*1是平移
#     # 输出 ： errRot, errPosition ： 旋转误差和平移误差
#     ###旋转误差的计算
#     rPred = preTrans[:, :3]
#     rGt = gtTrans[:, :3]
#     tt = np.transpose(rPred)
#     tmpR = np.dot(tt, rGt)
#     trace = tmpR[0][0] + tmpR[1][1] + tmpR[2][2]
#     if trace > 3:
#         trace = 3
#     elif trace < -1:
#         trace = -1
#     errRot = math.acos((trace - 1) / 2) * 180 / math.pi
#     ###平移的计算
#     tPred = preTrans[:, 3]
#     tGt = gtTrans[:, 3]
#     l22 = pow(tGt[0] - tPred[0], 2) + pow(tGt[1] - tPred[1], 2) + pow(tGt[2] - tPred[2], 2)
#     errPosition = math.sqrt(l22)
#     return errRot, errPosition
#
# #保存点云为ply文件
# def SavePoints_asPLY(points, path):
#     #points ： np类型点云n*3
#     #path ： 保存的路径，如： "/home/zzh/awsl-JL/object-deformnet-master/object-deformnet-master/data/results/"+"/points.ply"
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.io.write_point_cloud(path, pcd) #保存点云
# def save_KeyPoints(save_dir_root, orgPriorKey, orgInstanKey, noiInstanKey, svdInstanKey):
#     # name = str(cat_id)
#     save_dir = save_dir_root
#     os.makedirs(save_dir, exist_ok=True)
#
#     # save pointclouds
#     SavePoints_asPLY(orgPriorKey, os.path.join(save_dir, 'orgPriorKey.ply'))
#     SavePoints_asPLY(orgInstanKey, os.path.join(save_dir, 'orgInstanKey.ply'))
#     SavePoints_asPLY(noiInstanKey, os.path.join(save_dir, 'noiInstanKey.ply'))
#     SavePoints_asPLY(svdInstanKey, os.path.join(save_dir, 'svdInstanKey.ply'))
#
# #  对点云进行位姿变换
# #  输入 ： noiseT ： 4*4 np位姿； instanceKey ： n*3 np的点
# #  输出 ： noisePoints ： n*3 np的点
# def TranPointsofPose(noiseT, instanceKey):
#     noisePoints = noiseT[:3, :3].dot(instanceKey[inst].T) + noiseT[:3, 3, np.newaxis]   # 3*8的np
#     noisePoints = noisePoints.T
#     return noisePoints
#
#
#
#







# # 将类ICP封装起来, FSNet的
# class CatIcpNet(nn.Module):  # 图片提取特征
#
#     def __init__(self):
#         super(CatIcpNet, self).__init__()
#         # 网络初始化
#         self.keyFeatsNet = KeysFeatsExtra()  # 特征提取
#         # keyFeatsNet = nn.DataParallel(keyFeatsNet)
#         self.tEstiNet = EstiSt()  # 位移和尺寸估计
#         self.RGreenNet = RotGreen(k=6)  # 红色旋转轴的预测
#         self.RedNet = RotRed(k=6)  # 绿色旋转轴的预测
#
#     def forward(self, sourceKey, targetKey):
#         '''
#          sourceKey targetKey : tensor : batch * 8 * 3
#          sourceKey ： 类关键点； targetKey： 实例关键点
#          注意 ： 训练的时候batch不能为1！因为有batchNormal！！
#         '''
#
#         # 使用3DGCN提取特征
#         sourceFeats = self.keyFeatsNet(sourceKey)  # 64, 8, 768
#         targetFeats = self.keyFeatsNet(targetKey)
#         feats = torch.cat([sourceFeats, targetFeats], dim=2)  # 64, 8, 1536
#         # 估计平移
#         pts_s = targetKey.transpose(2, 1)
#         cen_pred, s = self.tEstiNet((pts_s - pts_s.mean(dim=2, keepdim=True)).float())  # [64, 3, 1]
#         T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)  ## 1x3x1
#         # 估计旋转
#         feavec = feats.transpose(1, 2)
#         RGreen = self.RGreenNet(feavec)  # [64, 6]
#         Red = self.RedNet(feavec)
#         return Red, RGreen, T_pred, s
#         '''
#          返回 ： Red， RGreen ： 两个旋转轴 tensor [bs, 6]
#          T_pred : 位移 [bs, 3, 1]
#          s ： 尺寸 [bs]
#         '''





    # UsingData = 'camera'  # camera : 仿真，real ： 实拍
    # BATCH_SIZE = 64
    # epochs = 50  # 50
    #
    # # train
    # if UsingData == 'camera':
    #     trainDataPath = ['data/KeyPointNOCS/camera_train.pkl']
    #     testDataPath = ['data/KeyPointNOCS/camera_val.pkl']
    # else :
    #     trainDataPath = ['data/KeyPointNOCS/camera_train.pkl', 'data/KeyPointNOCS/real_train.pkl']
    #     testDataPath = ['data/KeyPointNOCS/real_test.pkl']
    # # 读取全部的数据
    # trainData = DataLoad(trainDataPath)
    # testData = DataLoad(testDataPath)
    # # 把 dataset 放入 DataLoader
    # trainLoader = Data.DataLoader(dataset=trainData,  batch_size=BATCH_SIZE,  shuffle=True, num_workers=4)
    # testLoader = Data.DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=4)
    # # 网络初始化
    # keyFeatsNet = KeysFeatsExtra()
    # # keyFeatsNet = nn.DataParallel(keyFeatsNet)
    # keyFeatsNet = keyFeatsNet.train()
    # keyFeatsNet.cuda()
    #
    # tEstiNet = EstiSt()
    # tEstiNet = tEstiNet.train()
    # tEstiNet.cuda()
    #
    # RGreenNet = RotGreen(k=6)
    # RGreenNet = RGreenNet.train()
    # RGreenNet.cuda()
    # RedNet = RotRed(k=6)
    # RedNet = RedNet.train()
    # RedNet.cuda()
    #
    # # 优化器
    # lr = 0.001
    # optimizer = optim.Adam([{'params': keyFeatsNet.parameters()}, {'params': tEstiNet.parameters()},
    #                         {'params': RGreenNet.parameters()}, {'params': RedNet.parameters()}],
    #                        lr=lr, betas=(0.9, 0.99))
    #
    # # loss函数
    # Loss_func_ce = nn.MSELoss()
    # Loss_func_Rot1 = nn.MSELoss()
    # Loss_func_Rot2 = nn.MSELoss()
    #
    # for epoch in range(epochs):
    #     # 随着轮数的增加调节学习率
    #     if epoch > 0 and epoch % (epochs // 5) == 0:
    #         lr = lr / 4
    #     optimizer.param_groups[0]['lr'] = lr
    #     optimizer.param_groups[1]['lr'] = lr * 10
    #     optimizer.param_groups[2]['lr'] = lr * 20
    #     optimizer.param_groups[3]['lr'] = lr * 20
    #
    #     for step, data in enumerate(trainLoader):
    #         sourceKey, targetKey, gt, gtargetKey = data  # keys : batch*8*3
    #         sourceKey = sourceKey.cuda()
    #         targetKey = targetKey.cuda()
    #         gt = gt.cuda()
    #         gtargetKey = gtargetKey.cuda()
    #
    #         # 输入到网络
    #         # 使用3DGCN提取特征
    #         sourceFeats = keyFeatsNet(sourceKey)  # 64, 8, 768
    #         targetFeats = keyFeatsNet(targetKey)
    #         feats = torch.cat([sourceFeats, targetFeats], dim=2)  # 64, 8, 1536
    #         # 估计平移
    #         pts_s = targetKey.transpose(2, 1)
    #         cen_pred, s = tEstiNet((pts_s - pts_s.mean(dim=2, keepdim=True)).float())  # [64, 3, 1]
    #         T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)  ## 1x3x1
    #         # 估计旋转
    #         feavec = feats.transpose(1, 2)
    #         RGreen = RGreenNet(feavec)  #  [64, 6]
    #         Red = RedNet(feavec)
    #
    #         # 计算loss并更新参数
    #         optimizer.zero_grad()
    #         corners, centers = data_augment(targetKey.cpu().numpy(), gt[:, :3, :3].cpu(), gt[:, :3, 3].cpu())  # 其实这个计算标签的没怎么看懂，要认真读下
    #         centers = Variable(torch.Tensor((centers))).cuda()
    #         corners = Variable(torch.Tensor((corners))).cuda()
    #         loss_res = Loss_func_ce(cen_pred, centers.float())
    #         green_v = corners[:, 0:6].float().clone()
    #         red_v = corners[:, (0, 1, 2, 6, 7, 8)].float().clone()
    #         loss_rot_g = Loss_func_Rot1(RGreen, green_v)
    #         loss_rot_r = Loss_func_Rot2(Red, red_v)
    #         Loss = loss_res / 20.0 + loss_rot_r/100.0 + loss_rot_g/100.0
    #         Loss.backward()
    #         optimizer.step()
    #
    #         # 显示和保存loss
    #         trainlosses['loss_res'] = loss_res
    #         trainlosses['loss_rot_g'] = loss_rot_g
    #         trainlosses['loss_rot_r'] = loss_rot_r
    #         trainlosses['Loss'] = Loss
    #         writer.write_losses(trainlosses, step+epoch*len(trainLoader))
    #         if step % 10 == 0:
    #             print('loss_res : ', trainlosses['loss_res'], 'loss_rot_g : ', trainlosses['loss_rot_g'], 'loss_rot_r : ', trainlosses['loss_rot_r'], 'Loss : ', trainlosses['Loss'])
    #
    #
    #
    # # 验证
    # keyFeatsNet = keyFeatsNet.eval()
    # tEstiNet = tEstiNet.eval()
    # RedNet = RedNet.eval()
    # RGreenNet = RGreenNet.eval()
    # statisResults = {'10deg': 0, '10cm': 0, '10deg10cm': 0, '5deg': 0, '5cm': 0, '5deg5cm': 0, 'aveErrR': 0,
    #                  'aveErrt': 0}
    # for step, data in enumerate(testLoader):
    #     sourceKey, targetKey, gt, gtargetKey = data  # keys : batch*8*3
    #     sourceKey = sourceKey.cuda()
    #     targetKey = targetKey.cuda()
    #     gt = gt.cuda()
    #     gtargetKey = gtargetKey.cuda()
    #
    #     # 输入到网络
    #     # 使用3DGCN提取特征
    #     sourceFeats = keyFeatsNet(sourceKey)  # 64, 8, 768
    #     targetFeats = keyFeatsNet(targetKey)
    #     feats = torch.cat([sourceFeats, targetFeats], dim=2)  # 64, 8, 1536
    #     # 估计平移
    #     pts_s = targetKey.transpose(2, 1)
    #     cen_pred, s = tEstiNet((pts_s - pts_s.mean(dim=2, keepdim=True)).float())  # [64, 3, 1]
    #     T_pred = pts_s.mean(dim=2, keepdim=True) + cen_pred.unsqueeze(2)  ## 预测的平移
    #     # 估计旋转
    #     feavec = feats.transpose(1, 2)
    #     RGreen = RGreenNet(feavec)  # [64, 6]
    #     Red = RedNet(feavec)
    #
    #     # 误差统计和保存
    #     pred_axis = np.zeros((1, 3, 3))
    #     pred_axis[:, 0:2, :] = RGreen.view((1, 2, 3)).detach().cpu().numpy()
    #     pred_axis[:, 2, :] = Red.view((1, 2, 3)).detach().cpu().numpy()[:, 1, :]
    #     pred_axis[0] = pred_axis[0] / np.linalg.norm(pred_axis[0])  # 预测的旋转矩阵
    #     torch.cuda.empty_cache()
    #     # 计算误差
    #     corners0 = torch.Tensor(np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]]))
    #     corners0 = corners0.cuda()
    #     corners_ = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    #     kpm_gt = (trans_3d(corners_, gt[:, :3, :3].cpu(), np.array([0, 0, 0]).T).T).flatten()
    #     kpm_gt = kpm_gt.reshape((3, 3))
    #     kpm_gt = kpm_gt / np.linalg.norm(kpm_gt)
    #     cor0 = corners0.cpu().numpy()
    #     cor0 = cor0 / np.linalg.norm(cor0)
    #     pose_gt = gettrans(cor0.reshape((3, 3)), kpm_gt.reshape((3, 1, 3)))
    #     Rt = pose_gt[0][0:3, 0:3]
    #     pose = gettrans(cor0.reshape((3, 3)), pred_axis.reshape((3, 1, 3)))
    #     R = pose[0][0:3, 0:3]
    #
    #     print(type(gt[:, :3, 3].cpu().numpy()), type(T_pred))
    #     R_err, T_err = get6dpose1(Rt, gt[:, :3, 3].cpu().detach().numpy(), R, T_pred.cpu().detach().numpy())
    #     err = [R_err, T_err]
    #     print(R_err, T_err)
    #     # 统计误差
    #     if err[0] < 10:
    #         statisResults['10deg'] = statisResults['10deg'] + 1
    #     if err[1] < 0.1:
    #         statisResults['10cm'] = statisResults['10cm'] + 1
    #     if err[0] < 10 and err[1] < 0.1:
    #         statisResults['10deg10cm'] = statisResults['10deg10cm'] + 1
    #         # savePath = 'SelfResults/others/CatIcpNetSolve/'
    #         # TSVD = np.zeros((4, 4))
    #         # TSVD[:3, :3] = pred_axis[0]
    #         # TSVD[:3, 3] = T_pred[0].cpu().detach().numpy().reshape(-1)
    #         # print(TSVD)
    #         # invTSVD = np.linalg.inv(TSVD)
    #         # svdInstanKey = invTSVD[:3, :3].dot(targetKey.T) + invTSVD[:3, 3, np.newaxis]
    #         # svdInstanKey = svdInstanKey.T
    #         # save_KeyPoints(savePath + 'less10deg10cm', sourceKey, gtargetKey, targetKey, svdInstanKey)
    #         # asdf
    #     # else:
    #         # statisResults['10deg10cm'] = statisResults['10deg10cm'] + 1
    #         # savePath = 'SelfResults/others/CatIcpNetSolve/'
    #         # TSVD = np.zeros((4, 4))
    #         # TSVD[:3, :3] = pred_axis[0]
    #         # TSVD[:3, 3] = T_pred[0].cpu().detach().numpy().reshape(-1)
    #         # print(TSVD)
    #         # invTSVD = np.linalg.inv(TSVD)
    #         # svdInstanKey = invTSVD[:3, :3].dot(targetKey.T) + invTSVD[:3, 3, np.newaxis]
    #         # svdInstanKey = svdInstanKey.T
    #         # save_KeyPoints(savePath + 'more10deg10cm', sourceKey, gtargetKey, targetKey, svdInstanKey)
    #         # asdf
    #     if err[0] < 5:
    #         statisResults['5deg'] = statisResults['5deg'] + 1
    #     if err[1] < 0.05:
    #         statisResults['5cm'] = statisResults['5cm'] + 1
    #     if err[0] < 5 and err[1] < 0.05:
    #         statisResults['5deg5cm'] = statisResults['5deg5cm'] + 1
    #     statisResults['aveErrR'] = statisResults['aveErrR'] + err[0]
    #     statisResults['aveErrt'] = statisResults['aveErrt'] + err[1]
    #     # 统计
    # statisResults['10deg'] = statisResults['10deg'] / len(testData)
    # statisResults['10cm'] = statisResults['10cm'] / len(testData)
    # statisResults['10deg10cm'] = statisResults['10deg10cm'] / len(testData)
    # statisResults['5deg'] = statisResults['5deg'] / len(testData)
    # statisResults['5cm'] = statisResults['5cm'] / len(testData)
    # statisResults['5deg5cm'] = statisResults['5deg5cm'] / len(testData)
    # statisResults['aveErrR'] = statisResults['aveErrR'] / len(testData)
    # statisResults['aveErrt'] = statisResults['aveErrt'] / len(testData)
    # print(statisResults)
    #     # asdf
    #
    #
    #
