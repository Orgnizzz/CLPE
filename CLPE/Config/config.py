from __future__ import print_function

import absl.flags as flags




#########################写入tensorboard##############################
flags.DEFINE_integer('writerFlag', 1, '是否写入tensorBoard')
flags.DEFINE_string('writerPath', 'SelfResults/Logs/tmpWriter', 'path to save checkpoint')
###################发邮件################
flags.DEFINE_integer('sendEmailFlag', 0, '发不发邮件')

# datasets
flags.DEFINE_integer('obj_c', 6, 'nnumber of categories')
flags.DEFINE_string('dataset', 'Real', 'CAMERA or Real')  # 用的是real
flags.DEFINE_string('dataset_dir', '/data1/jl/GPVNet/data/NOCS/data', 'path to the dataset')
flags.DEFINE_string('detection_dir', '/data1/jl/GPVNet/data/NOCS/NOCS_segmentation_results', 'path to detection results')
flags.DEFINE_string('per_obj', '', 'only train an specified object')
# train parameters
# train##################################################
flags.DEFINE_integer("train", 1, "1 for train mode")
# flags.DEFINE_integer('eval', 0, '1 for eval mode')
flags.DEFINE_string('device', 'cuda:0, 1, 2, 3', '')
# flags.DEFINE_string("train_gpu", '0', "gpu no. for training")
flags.DEFINE_integer("num_workers", 8, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size', 24, '')
flags.DEFINE_integer('s_epoch', 0, '开始的轮数')  # 150
flags.DEFINE_integer('total_epoch', 300, 'total epoches in training')  # 150
flags.DEFINE_integer('train_steps', 1000, 'number of batches in each epoch')  # batchsize is 8, then 3000
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate

# dynamic zoom in
flags.DEFINE_float('DZI_PAD_SCALE', 1.5, '')
flags.DEFINE_string('DZI_TYPE', 'uniform', '')
flags.DEFINE_float('DZI_SCALE_RATIO', 0.25, '')
flags.DEFINE_float('DZI_SHIFT_RATIO', 0.25, '')

# input parameters
flags.DEFINE_integer("img_size", 256, 'size of the cropped image')
# data aug parameters
flags.DEFINE_integer('roi_mask_r', 3, 'radius for mask aug')
flags.DEFINE_float('roi_mask_pro', 0.5, 'probability to augment mask')
flags.DEFINE_float('aug_pc_pro', 0.2, 'probability to augment pc')
flags.DEFINE_float('aug_pc_r', 0.05, 'max change 5% of the point')
flags.DEFINE_float('aug_rt_pro', 0.3, 'probability to augment rt')
flags.DEFINE_float('aug_bb_pro', 0.3, 'probability to augment size')
flags.DEFINE_float('aug_bc_pro', 0.3, 'box cage based augmentation, only valid for bowl, mug')

#
flags.DEFINE_integer('num_points', 500, 'points')
flags.DEFINE_integer('num_cates', 6, 'number of categories')
flags.DEFINE_integer('num_kp', 8, 'number of kp')

#学习率
flags.DEFINE_float('lr', 0.0001, 'learning rate')


###################使用的loss函数########################
flags.DEFINE_integer('simCatLossFlag', 0, '是否使用模型相似度loss')
flags.DEFINE_integer('L1KeysLossFlag', 1, '是否使用关键点正则化loss')  # 使关键点贴近物体表面
flags.DEFINE_integer('sRtLossFlag', 1, '是否使用PoseSizeloss')
flags.DEFINE_integer('bboxNorLossFlag', 1, '是否使用位姿增强的loss，bbox法向')
# loss的权重
flags.DEFINE_float('catSimLossWeight', 8.0, '相似度loss的权重')  # 1
flags.DEFINE_float('l1KeysLossWeight', 8.0, '关键点L1loss的权重')
# 类icpLoss的权重
flags.DEFINE_float('rot_1_w', 8.0, '')
flags.DEFINE_float('rot_2_w', 8.0, '')
flags.DEFINE_float('rot_regular', 4.0, '')
flags.DEFINE_float('tran_w', 8.0, '')
flags.DEFINE_float('size_w', 8.0, '')
flags.DEFINE_float('r_con_w', 1.0, '')

# 先用关键点初始化的loss训练的轮数
# 一共训练300轮，关键点训150， poseSize训150，其中100轮没有bbox法向约束，后50轮有bbox法向约束
flags.DEFINE_integer('iterations_init_points', 150, '先训练关键点正则化的轮数')
flags.DEFINE_integer('iterations_fineTuningPose', 250, '训练位姿收敛后，加上包围盒法向的loss进行微调')


###################网络预训练模型保存路径########################
flags.DEFINE_string('model_save', '/data1/jl/6pack/SelfResults/Pths/tmp/', '网络预训练模型保存路径')

###################恢复预训练模型的路径########################
# 恢复全部网络的
flags.DEFINE_integer('resume', 0, '是否恢复网络的预训练模型')
flags.DEFINE_string('resume_model', '/data1/jl/6pack/SelfResults/Pths/tmp/54net.pth', '恢复网络预训练模型的路径')
# 恢复部分网络的，这个只能恢复关键点网路的模型
flags.DEFINE_integer('resumePart', 0, '是否恢复部分网络的预训练模型')
flags.DEFINE_string('resumePart_model', '/data1/jl/6pack/SelfResults/Pths/tmp/54net.pth', '恢复部分网络预训练模型的路径')

###############是否做数据增强##########################
flags.DEFINE_integer('doAugFlag', 0, '是否做数据增强')


