import os
import torch
import random
from Network.Network import KeysNetwork
from Config.config import *
from absl import app

FLAGS = flags.FLAGS
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import open3d as o3d

from DataLoad.load_data_eval import PoseDataset
from Utils.evaluate_utils.eval_utils import setup_logger, compute_mAP
from Utils.evaluate_utils.eval_utils_v1 import compute_degree_cm_mAP
from Utils.evaluate_utils.geom_utils import generate_RT, generate_sRT
from Utils.utils import load_network

device = 'cuda'

def NormDisF2Point(viewpoints, normal, dis, f):  # 根据观测点云的方向、距离和置信度计算到6个面的投影
    # 输入 ： tensor normal： bs*num*6*3; dis : bs*num*6; f: bs*num*6;
    keyPoints_f_num = torch.sum(f, dim=1).unsqueeze(1)  # 分母，用f除，即得到每个点的权重
    keyPoints_c = f / keyPoints_f_num  # 权重
    # print(keyPoints_c, keyPoints_c.sum())
    # asdf

    pointPreject = (viewpoints).unsqueeze(2) + (normal * ((dis).unsqueeze(-1)))
        # (dis * keyPoints_c).unsqueeze(-1)))  # 24, 1028, 8, 3 因为对每个点都指出了关键点，此时得到的应该是以8个关键点为中心的一堆点
    return pointPreject

# 计算两向量的夹角
def angle2(v1,v2):
    x=np.array(v1)
    y=np.array(v2)
    # 分别计算两个向量的模：
    module_x=np.sqrt(x.dot(x))
    module_y=np.sqrt(y.dot(y))
    # 计算两个向量的点积
    dot_value=x.dot(y)
    # 计算夹角的cos值：
    cos_theta=dot_value/(module_x*module_y)
    # 求得夹角（弧度制）：
    angle_radian=np.arccos(cos_theta)
    # 转换为角度值：
    angle_value=angle_radian*180/np.pi
    return angle_value
def evaluate(argv):
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    # tf.compat.v1.disable_eager_execution()
    FLAGS.train = False

    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    # build dataset annd dataloader

    val_dataset = PoseDataset(source=FLAGS.dataset, mode='test')
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    if os.path.exists(pred_result_save_path):
        print('exist the same result, please delete it')
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
    else:
        network = KeysNetwork(FLAGS).cuda()
        network = network.to(device)

        if FLAGS.resume:
            print('loading PreNetModel'+FLAGS.resume_model)
            load_network(network, FLAGS.resume_model)

            # state_dict = torch.load(FLAGS.resume_model)
            # network.NetResume(state_dict)
        # else:
        #     raise NotImplementedError
        # start to test
        print('start to test')
        network = network.eval()
        pred_results = []
        for i, data in tqdm(enumerate(val_dataset, 1)):
            if data is None:
                print('data is None')
                continue
            data, detection_dict, gts = data
            mean_shape = data['mean_shape'].to(device)
            sym = data['sym_info'].to(device)
            if len(data['cat_id_0base']) == 0:
                detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
                detection_dict['pred_scales'] = np.zeros((0, 4, 4))
                pred_results.append(detection_dict)
                # print('len(data[cat_id_0base]) == 0')
                continue

            # 网络预测
            output_dict = network(data['roi_img'].cuda(), data['choose'].cuda(), data['PC'].cuda(), data['catKeys'].cuda(), data['cat_id_0base'].cuda(),
                                  trainFlag=0)

            # 输出关键点
            '''  
            ###############################保存输出#############################################
            # 试图输出关键点和观测点云
            # 预测的实例关键点inCam、观测点云、gt类和实例关键点inNOCS
            keyPoints = output_dict['preInsKeys']
            pre_instanKeyPts = keyPoints.detach()
            viewPoints = data['PC'].detach()
            gt_catKeys = data['catKeys']
            gt_insKeys = data['insKeys']
            # 通过pred_class_ids找gtid的位置，进而找到gt的pose和size
            pred_class_ids = detection_dict['pred_class_ids']
            gtClassIdx = np.argwhere(detection_dict['gt_class_ids'] == pred_class_ids)
            gtsRT = detection_dict['gt_RTs'][gtClassIdx]  # 这个里面的旋转已经嵌入了尺寸！
            # gtS = data['gtSone'][str(pred_class_ids[0])]   # 直接读取尺寸
            # 在gtInstance中找到对应的模型
            cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
            frameCat = cat_names[pred_class_ids[0]-1]
            for key in gt_insKeys:
                if key.find(frameCat) == 0:
                    cateIndex = key

            # 将NOCS下的类和实例关键点通过9D变换到相机下
            s1 = np.cbrt(np.linalg.det(gtsRT[0, 0, :3, :3]))  # 这个就是正确的尺寸
            gt_catKeysinCam = TranPointsofPose(gtsRT[0, 0], gt_catKeys[0])
            gt_insKeysinCam = TranPointsofPose(gtsRT[0, 0], gt_insKeys[cateIndex])

            # # 保存
            # # SavePoints_asPLY(gt_catKeys[0],
            # #                  '/data1/jl/6pack/SelfResults/others/ProveinTest/' + str(i) + 'orgCateKeys.ply')
            # # SavePoints_asPLY(gt_insKeys[cateIndex],
            # #                  '/data1/jl/6pack/SelfResults/others/ProveinTest/' + str(i) + 'orgInsKeys.ply')
            # print(viewPoints[0].shape)
            # SavePoints_asPLY(viewPoints[0].cpu().numpy(), '/data1/jl/6pack/SelfResults/others/ProveinTest/'+str(i)+'points.ply')
            # SavePoints_asPLY(pre_instanKeyPts[0].cpu().numpy(),
            #                  '/data1/jl/6pack/SelfResults/others/ProveinTest/'+str(i)+'pre_insKeysinCam.ply')
            # SavePoints_asPLY(gt_catKeysinCam, '/data1/jl/6pack/SelfResults/others/ProveinTest/'+str(i)+'gt_catKeysinCam.ply')
            # SavePoints_asPLY(gt_insKeysinCam, '/data1/jl/6pack/SelfResults/others/ProveinTest/'+str(i)+'gt_insKeysinCam.ply')

            test2ViewandKey2NOCSFlag = 1  # 将观测点云和预测的关键点投影到NOCS中并保存
            if test2ViewandKey2NOCSFlag == 1:
                # 将预测的关键点和观测点云反变换到NOCS空间中判断一致性
                s1 = np.cbrt(np.linalg.det(gtsRT[0, 0, :3, :3]))  # 这个就是正确的尺寸
                s1 = torch.tensor(s1)
                gtRt = gtsRT.copy()[0]
                gtRt[:, :3, :3] = gtRt[:, :3, :3]/s1
                gtRt = torch.tensor(gtRt)
                print('11111111111111111111111111111', i)
                viewPtsinNOCS = InvTransform9D(viewPoints, s1.unsqueeze(0), gtRt)
                preKeysinNOCS = InvTransform9D(pre_instanKeyPts.cpu().detach(), s1.unsqueeze(0), gtRt)

                SavePoints_asPLY(viewPtsinNOCS[0].numpy(), '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTest/epoch294/'+str(i)+'viewPtsinNOCS.ply')
                SavePoints_asPLY(preKeysinNOCS[0].numpy(), '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTest/epoch294/'+str(i)+'preKeysinNOCS.ply')
                SavePoints_asPLY(gt_insKeys[cateIndex],
                                 '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTest/epoch294/' + str(i) + 'orgInsKeys.ply')
            '''
            ###############################统计误差#############################################
            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()
            from Utils.utils import get_gt_v
            pred_s = p_s #+ mean_shape

            # print(mean_shape, pred_s)

            # print(p_green_R_vec.shape, p_red_R_vec.shape, p_T.shape)
            # asdf
            pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

            if pred_RT is not None:
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_s = pred_s.detach().cpu().numpy()
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_s
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)

        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

        # if FLAGS.eval_inference_only:
        #     import sys
        #     sys.exit()

        degree_thres_list = list(range(0, 61, 1))
        shift_thres_list = [i / 2 for i in range(21)]
        iou_thres_list = [i / 100 for i in range(101)]

        # iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
        #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
        synset_names = ['BG'] + ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        if FLAGS.per_obj in synset_names:
            idx = synset_names.index(FLAGS.per_obj)
        else:
            idx = -1
        iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list,
                                                  shift_thres_list,
                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True, )

        # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
        iou_25_idx = iou_thres_list.index(0.25)
        iou_50_idx = iou_thres_list.index(0.5)
        iou_75_idx = iou_thres_list.index(0.75)
        degree_05_idx = degree_thres_list.index(5)
        degree_10_idx = degree_thres_list.index(10)
        shift_02_idx = shift_thres_list.index(2)
        shift_05_idx = shift_thres_list.index(5)
        shift_10_idx = shift_thres_list.index(10)

        messages = []

        if FLAGS.per_obj in synset_names:
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
        else:
            messages.append('average mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

            for idx in range(1, len(synset_names)):
                messages.append('category {}'.format(synset_names[idx]))
                messages.append('mAP:')
                messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
                messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
                messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
                messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
                messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
                messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
                messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
                messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

        for msg in messages:
            logger.info(msg)


#  对点云进行位姿变换
#  输入 ： noiseT ： 4*4 np位姿； instanceKey ： n*3 np的点
#  输出 ： noisePoints ： n*3 np的点
def TranPointsofPose(noiseT, instanceKey):
    noisePoints = noiseT[:3, :3].dot(instanceKey.T) + noiseT[:3, 3, np.newaxis]   # 3*8的np
    noisePoints = noisePoints.T
    return noisePoints
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

def SavePoints_asPLY(points, path):
    #points ： np类型点云n*3
    #path ： 保存的路径，如： "/home/zzh/awsl-JL/object-deformnet-master/object-deformnet-master/data/results/"+"/points.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd) #保存点云


if __name__ == "__main__":
    # # 查看pkl文件
    # import _pickle as cPickle
    # import pickle
    # import pprint
    # file = open("/data1/jl/GPVNet/data/NOCS/KeyPoints/real_test.pkl", "rb")
    # data = pickle.load(file)
    # pprint.pprint(data)
    #
    #
    # file = open('/data1/jl/awsl-JL/object-deformnet-master/object-deformnet-master/data/Real/test/scene_1/0304_label.pkl', "rb")
    # data = pickle.load(file)
    # pprint.pprint(data)
    # asdf
    # 将各类的关键点单独保存
    # keyPoints = {}
    # with open("/data1/jl/GPVNet/data/NOCS/KeyPoints/real_train.pkl", 'rb') as f:
    #     keyPoints.update(cPickle.load(f))
    #
    # # 分开类关键点和实例关键点；用字典存储
    # categoryKey = {}
    # instanceKey = {}
    # for key in keyPoints.keys():
    #
    #     judgeKey = key.split('_source')
    #     if len(judgeKey) == 2:
    #         categoryKey[judgeKey[0]] = keyPoints[key]
    #     else:
    #         judgeKey = key.split('_target')
    #         instanceKey[judgeKey[0]] = keyPoints[key]
    # print(categoryKey)
    #
    # categoryKeys = {}
    # cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    # for key in categoryKey:
    #     cate = key.split('_')[0]
    #     if cate in cat_names:
    #         categoryKeys[cate] = categoryKey[key]
    # print('sum : ', categoryKeys)
    # with open('/data1/jl/GPVNet/data/NOCS/KeyPoints/CategoryKeyPoints.pkl', "wb") as f:
    #     pickle.dump(categoryKeys, f)
    # asdf



    app.run(evaluate)