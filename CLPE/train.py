import os
import random
import math
import torch
from absl import app
import numpy as np
import time
import torch.nn as nn

from SelfUtils import SendEmail
from SelfUtils import SumWriter
from DataLoad.load_data import PoseDataset
from Network.Network import KeysNetwork
from Config.config import *
FLAGS = flags.FLAGS
torch.autograd.set_detect_anomaly(True)
device = 'cuda'

from SelfUtils import SumWriter  #头文件
from Utils.utils import save_network, load_network
from Utils.utils import transform9D, SavePoints_asPLY, InvTransform9D
from Utils.DataAug_RBPose import data_augment

def train(argv):
    # if FLAGS.writerFlag == 1:
    writer = SumWriter.save_summary(FLAGS.writerPath)
    if FLAGS.sendEmailFlag == 1:
        SE = SendEmail.SendEmail()  # 在开头实例一个发邮件的

    # 数据集
    s_epoch = 0
    train_dataset = PoseDataset(source=FLAGS.dataset, mode='train',  # 读取数据
                                data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_steps
    global_step = train_steps * s_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    #  网络
    net = KeysNetwork(FLAGS).cuda()
    if FLAGS.resume == 1:
        load_network(net, FLAGS.resume_model)
    if FLAGS.resumePart == 1:
        state_dict = torch.load(FLAGS.resumePart_model)
        net.NetResume(state_dict)
    net.train()

    for epoch in range(FLAGS.s_epoch, FLAGS.total_epoch):
        # train one epoch
        # create optimizer and adjust learning rate accordingly
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if FLAGS.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len + real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3 * n_repeat * real_len) + real_indices * n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)

        #################################
        for i, data in enumerate(train_dataloader, 1):
            # 根据RBPPose做数据增强
            if FLAGS.doAugFlag > 0:
                print(data['sym_info'])
                awd
                PC_da, gt_R_da, gt_t_da, gt_s_da, model_point, PC_nocs = data_augment(data['PC'], data['rotation'], data['translation'], data['fsnet_scale'],
                                                                                      data['mean_shape'],
                                                                                      data['sym_info'],
                                                                                      data['aug_bb'],
                                                                                      data['aug_rt_t'], data['aug_rt_R'], data['model_point'],
                                                                                      data['nocs_scale'], data['insKeys'],
                                                                                      data['cat_id'])

                # 数据增强里直接修改了原数据； tensor想拷贝应该加clone()，不然数据会随着发生改变
                data['PC'] = PC_da
                data['rotation'] = gt_R_da
                data['translation'] = gt_t_da
                data['fsnet_scale'] = gt_s_da
                data['model_point'] = model_point
                data['insKeys'] = PC_nocs
                # 处理nocs_scale的增强问题
                data['nocs_scale'] = torch.norm(data['mean_shape'] + data['fsnet_scale'], dim=1)

            # 网络预测
            output = net(data['roi_img'].cuda(), data['choose'].cuda(), data['PC'].cuda(), data['catKeys'].cuda(),
                         data['cat_id'].cuda(), data['catModel'].cuda(), data['mean_shape'].cuda())  # 第二行的参数，是变形相关网络需要的信息

            # 类实例模型，预测的控制点，转到NOCS空间，输出保存
            ''' 
            # 将观测点云，预测的关键点，实例关键点inCam pre指向实例关键点，保存输出
            # 将tensor类型的Rt合成T
            bs = data['rotation'].shape[0]
            gtRt = torch.zeros([bs, 3, 4], device=data['insKeys'].device)  # 新建到相同的device上
            gtRt[:, :3, :3] = data['rotation']
            gtRt[:, :3, 3] = data['translation']
            preInsinCam = InvTransform9D(output['preInsKeys'].cpu().detach(), data['nocs_scale'], gtRt)
            outInsPts = data['model_point'][0].cpu().numpy()
            outPts = data['catModel'][0].cpu().numpy()
            outPre = preInsinCam[0].numpy()
            outGtCate = data['catKeys'][0].cpu().detach().numpy()
            outGtIns = data['insKeys'][0].cpu().detach().numpy()
            # 将观测点云变化到NOCS空间
            viewPts = InvTransform9D(data['PC'], data['nocs_scale'], gtRt)
            viewPts = viewPts[0].numpy()
            # 计算预测指向gt的方向
            outGt = outGtIns
            normals = outGt - outPre
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / norm
            outNormals = normals
            # 输出为ply
            # SavePoints_asPLY(outInsPts, '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTrain/epoch299/' + str(i) + 'insModel.ply')
            # SavePoints_asPLY(outPts, '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTrain/epoch299/'+str(i)+'cateModel.ply')
            SavePoints_asPLY(viewPts, '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTrain/epoch299/' + str(i) + 'viewPts.ply')
            SavePoints_asPLY(outPre, '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTrain/epoch299/'+str(i)+'preKeys.ply', outNormals)
            SavePoints_asPLY(outGtIns, '/data1/jl/cate9DPoseSim2Real/SelfResults/others/ProveInTrain/epoch299/'+str(i)+'gtKeys.ply')
            print(i)
            '''

            # loss计算
            loss_data = {
                's': data['nocs_scale'].cuda(),
                'R': data['rotation'].cuda(),
                't': data['translation'].cuda(),
                'catModel': data['catModel'].cuda(),
                'catKeys': data['catKeys'].cuda(),
                'insKeys': data['insKeys'].cuda(),  # gt instance keys in NOCS
                'catInfluence': data['catInfluence'].cuda(),
                'insModel': data['model_point'].cuda(),
                'sym': data['sym_info'].cuda(),
                'gt_lwh': (data['fsnet_scale']+data['mean_shape']).cuda(),  # 真实的长宽高包围盒
            }

            losses = net.compute_loss(loss_data['s'], loss_data['R'], loss_data['t'], \
                             loss_data['catModel'], loss_data['catKeys'], loss_data['insKeys'], loss_data['catInfluence'], loss_data['insModel'], loss_data['sym'], loss_data['gt_lwh'])
            # 优化回传
            net.optimize(losses, epoch)

            # 保存训练的loss
            writer.write_losses(losses, i + epoch * len(train_dataloader))  # 在每步迭代中保存训练的loss

            # 打印loss
            if i%100 == 0:
                print('l1KeysLoss : ', losses['l1KeysLoss'].cpu().detach().numpy(), 'poseSiseLoss : ', losses['poseSiseLoss'].cpu().detach().numpy())

        # 保存网络训练模型
        # save_network(net, FLAGS.model_save, network_label="net")

        defaultEpochSave = [FLAGS.iterations_init_points-11, FLAGS.iterations_init_points-6, FLAGS.iterations_init_points-1, \
                            FLAGS.total_epoch-21, FLAGS.total_epoch-16, FLAGS.total_epoch-11, FLAGS.total_epoch-6, FLAGS.total_epoch-1]
        if epoch in defaultEpochSave:
            save_network(net, FLAGS.model_save, network_label="net", epoch_label=str(epoch))
        else:
            save_network(net, FLAGS.model_save, network_label="net", epoch_label='tmp')

        # net.NetSave(FLAGS.model_save, epoch)

    if FLAGS.sendEmailFlag == 1:
        SE.EndSend('poseSiseLoss', str(losses['poseSiseLoss'].cpu().detach().numpy()))  # 训练完发送



if __name__ == "__main__":
    # argv = 0
    # train(argv)
    app.run(train)

