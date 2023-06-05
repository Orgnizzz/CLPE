# follow FS-Net
import torch.nn as nn
import torch
import torch.nn.functional as F

import Utils.PreKeysNet_GPV_Utils.gcn3d as gcn3d

# global feature num : the channels of feature from rgb and depth
# grid_num : the volume resolution

class KeyNet(nn.Module):  # 这个应该是对应文章中的，用3DGCN提取特征，然后重建点云；还有对每个点到6个面的方向向量、距离和置信度
    def __init__(self):
        super(KeyNet, self).__init__()
        gcn_n_num = 10  # 'neighbor number for gcn'
        gcn_sup_num = 7  # 'support number for gcn'
        face_recon_c = 6 * 5  # 'for every point, we predict its distance and normal to each face'
        obj_c = 6  # 'number of categories'
        feat_face = 768  # 'input channel of the face recon'
        keyPointsNum = 8  # '关键点的数量'
        self.obj_c = obj_c
        self.keyPointsNum = keyPointsNum


        self.neighbor_num = gcn_n_num
        self.support_num = gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.recon_num = 3
        self.face_recon_num = face_recon_c

        dim_fuse = sum([128, 128, 256, 256, 512, obj_c])
        # 16: total 6 categories, 256 is global feature
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # 投票获得关键点的网络
        self.keyRecon = nn.Sequential(
            nn.Conv1d(feat_face + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, int(keyPointsNum*5), 1),  # Relu or not?  #到每个关键点的方向：3个， 距离1个、置信度1个，每个关键点5个参数，8个关键点，40个参数
        )

    def forward(self,
                pts: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)"
                ):
        # 输入 ： 观测点云和类别
        """
        pts ： 观测点云
        Return: (bs, vertice_num, class_num)
        """
        vertices = pts-pts.mean(dim=1, keepdim=True)
        #  concate feature
        bs, vertice_num, _ = vertices.size()
        # cat_id to one-hot
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, self.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        # bs x verticenum x 6

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0]  # (bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)  # 到这儿，往上都是提取观测点云特征的，feat是提取的最后的特征
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''
        # feat_face_re = self.global_perception_head(feat)  # bs x C x 1
        feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1)
        # feat is the extracted per pixel level feature

        conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)

        # average pooling for face prediction
        feat_face_in = torch.cat([feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1)


        # 加上关键点的预测
        #
        keyPoints = self.keyRecon(feat_face_in)  # batch*40(kpts*5)*1028
        keyPoints = keyPoints.permute(0, 2, 1)

        # 使用l1loss获得关键点
        insKeyPoints = keyPoints[:, :, :int(self.keyPointsNum) * 3].view(bs, vertice_num, int(self.keyPointsNum),
                                                                          3)  # 将特征处理成到每个点的方向向量 24, 1028, 8, 3
        insKeyPointsinCentor = insKeyPoints[:, 0]

        insKeyPoints = insKeyPointsinCentor + pts.mean(dim=1, keepdim=True)

        return insKeyPoints





# 提取特征, 提取关键点特征
class KeysFeatsExtra(nn.Module):
    def __init__(self, class_num=2, vec_num=1, support_num=7, neighbor_num=3):
        super(KeysFeatsExtra, self).__init__()
        self.neighbor_num = neighbor_num  # 这个应该是设置3Dgcn找一个点附近的几个点的

        self.conv_0 = gcn3d.Conv_surface(kernel_num=128, support_num=support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num=support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=2, neighbor_num=2)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num=support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num=support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num=support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        dim_fuse = sum([128, 128, 256, 256, 512, 512, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, class_num + vec_num * 3, 1),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 2))

        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        # v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
        #                                                         v_pool_2.shape[1] // 2))
        # fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        # f_global = fm_4.max(1)[0]  # (bs, f)
        #
        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        # nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        # fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3], dim=2)

        return feat




if __name__ == "__main__":
    print(1)
    points = torch.rand(2, 1024, 3)  # 先验模型
    obj_id = torch.Tensor([3, 4])  # 类别

    testNet = KeyNet()
    insKeyPointsinCentor = testNet(points - points.mean(dim=1, keepdim=True),
                                                                         obj_id)  # 把关键点预测作为可选项加到重建的里面了
    instanKeyPts = insKeyPointsinCentor + points.mean(dim=1, keepdim=True)
    print(instanKeyPts.shape)


