import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Utils.SPDResNorLoss_RBPose_Utils.Loss_each import shape_prior_loss, prop_rot_loss, recon_6face_loss


#################将RBPPose中，重建和对应，投影约束旋转平移，bbox法向残差约束写到一起############################
class BboxGeometryLoss(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self):  # 初始化的时候，给loss的权重
        super(BboxGeometryLoss, self).__init__()
        self.shapeLoss = shape_prior_loss()  # 重建实例模型和对应关系的约束
        self.rotTranPtsLoss = prop_rot_loss()  # 对观测做pre和gt位姿变换做的约束(应该是对纯位姿约束的加强)
        self.bboxResNorLoss = recon_6face_loss()  # bbox法向预测的约束

    def forward(self, sym,
                pred_list, gt_list,
                obj_ids, save_path=None,
                resBboxNorFlag=0):
        namelist = ["Prop_pm", 'Prop_point_cano', "Prop_sym",   # enhancePoseLoss
                    'Per_point']  # resBboxNorLoss

        assign_mat = pred_list['assign_mat']
        deltas = pred_list['deltas']
        prior = gt_list['prior']
        nocs = gt_list['nocs']
        model = gt_list['model']
        shapeCorLoss = self.shapeLoss(assign_mat, deltas, prior, nocs, model, sym)
        enhancePoseLoss = self.rotTranPtsLoss(namelist, pred_list, gt_list, sym)
        if resBboxNorFlag == 1:  # 先训练其他的，都收敛了再训练这个，给这个loss打开
            resBboxNorLoss = self.bboxResNorLoss(namelist, pred_list, gt_list, sym, obj_ids, save_path)
        else:
            resBboxNorLoss = 0
        # print(shapeCorLoss, enhancePoseLoss, resBboxNorLoss)
        return shapeCorLoss, enhancePoseLoss, resBboxNorLoss


if __name__ == "__main__":
    # 初始化一个loss
    testLoss = BboxGeometryLoss()

    # 生成测试数据
    obj_ids = torch.Tensor([3, 4])  # 类别
    sym = torch.rand(2, 4)
    #pre
    assign_mat = torch.rand(2, 8, 1024)
    deltas = torch.rand(2, 1024, 3)

    Rot1 = torch.rand(2, 3)
    Rot2 = torch.rand(2, 3)
    Rot1_f = torch.rand(2)
    Rot2_f = torch.rand(2)
    Tran = torch.rand(2, 3)
    Recon = torch.rand(2, 8, 3)

    face_shift = torch.rand(2, 8, 18)
    face_shift_delta = torch.rand(2, 8, 18)
    F_log_var = torch.rand(2, 8, 6)
    # gt
    prior = torch.rand(2, 1024, 3)
    nocs = torch.rand(2, 8, 3)
    model = torch.rand(2, 1024, 3)
    Points = torch.rand(2, 8, 3)
    R = torch.rand(2, 3, 3)
    T = torch.rand(2, 3)
    Size = torch.rand(2, 3)


    # 给到网络
    pred_list = {
        # shapeCorLoss 需要的数据
        'assign_mat': assign_mat,  #[bs, 8, 1024]
        'deltas': deltas,  # [bs, 1024, 3]

        # enhancePoseLoss 需要的数据
        'Rot1': Rot1,
        'Rot1_f': Rot1_f.detach(),
        'Rot2': Rot2,
        'Rot2_f': Rot2_f.detach(),
        'Tran': Tran,
        'Recon': Recon,

        # resBboxNorLoss 需要的数据
        'face_shift': face_shift,
        'face_shift_delta': face_shift_delta,
        'F_log_var': F_log_var,

    }

    gt_list = {
        # shapeCorLoss 需要的数据
        'prior': prior,  # 先验模型
        'nocs': nocs,  # 关键点在NOCS下的3D坐标
        'model': model,

        # enhancePoseLoss 需要的数据D
        'Points': Points,  # 预测的关键点inCam
        'R': R,
        'T': T,

        # resBboxNorLoss 需要的数据
        'Size': Size
    }
    testLoss(sym, pred_list, gt_list, obj_ids, resBboxNorFlag=1)

    # # prop loss
    # pred_prop_list = {
    #     'Recon': p_recon,
    #     'Rot1': p_green_R,
    #     'Rot2': p_red_R,
    #     'Tran': p_T,
    #     'Scale': p_s,
    #     'Rot1_f': f_green_R.detach(),
    #     'Rot2_f': f_red_R.detach(),
    # }
    #
    # gt_prop_list = {
    #     'Points': PC,
    #     'R': gt_R,
    #     'T': gt_t,
    #     'Point_mask': point_mask_gt
    # }
    # prop_loss = self.loss_prop(self.name_prop_list, pred_prop_list, gt_prop_list, sym)
    #
    # pred_recon_list = {
    #     'face_shift': face_shift,
    #     'face_shift_delta': face_shift_delta,
    #     'face_shift_prior': face_shift_prior,
    #     'F_log_var': face_log_var,
    #     'Pc_sk': PC_sk,
    #     'Rot1': p_green_R,
    #     'Rot1_f': f_green_R.detach(),
    #     'Rot2': p_red_R,
    #     'Rot2_f': f_red_R.detach(),
    #     'Tran': p_T,
    #     'Size': p_s,
    #     'Point_mask_conf': point_mask_conf
    # }
    # gt_recon_list = {
    #     'R': gt_R,
    #     'T': gt_t,
    #     'Size': gt_s,
    #     'Points': PC,
    #     'Point_mask': point_mask_gt
    # }
    #
    # if FLAGS.eval_visualize_pcl:
    #     save_path = os.path.join(FLAGS.model_save, 'voting_visual')
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_path = os.path.join(save_path, str(batch_num))
    # else:
    #     save_path = None
    # recon_loss = self.loss_recon(self.name_recon_list, pred_recon_list, gt_recon_list, sym, obj_id,
    #                              save_path=save_path)