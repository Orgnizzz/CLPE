import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops import rearrange
import numpy as np



from Utils.LossCatSim_keyPtsDefor_utils.cages import deform_with_MVC
from Utils.LossCatSim_keyPtsDefor_utils.utils import normalize_to_box, sample_farthest_points



class CatSimLoss(nn.Module):
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
        return rearrange

    
    def __init__(self):
        super(CatSimLoss, self).__init__()
        
        template_vertices, template_faces = self.create_cage()  #新建单位求笼，输出顶点和面
        self.init_template(template_vertices, template_faces)



    def create_cage(self):
        ico_sphere_div = 1
        # cage (1, N, 3)
        cage_size = 1.4
        mesh = pytorch3d.utils.ico_sphere(ico_sphere_div, device='cuda:0')  # 为单位球创建顶点和面，第一个参数是迭代次数，第二个是设备

        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F


    def init_template(self, template_vertices, template_faces):
        n_keypoints = 8
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(n_keypoints, self.template_vertices.shape[2]), requires_grad=True)


    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    
    def forward(self, catModel, catKeys, catInflu, insKeys, insModel):
        #  输入应该是 ： 类模型、类关键点、influence、实例关键点inNOCS、实例模型
        #  实例关键点inNOCS ： 是网络预测的实例关键点，经过RTs反变换到NOCS空间的
        #  输出应该是 ： 实例模型和deformed类模型的相似度
        """
        catModel : tensor B,num,3
        catKeys : tensor B,8,3  8个关键点
        catInflu ： tensor B,8,42
        insKeys ： tensor B,8,3  8个关键点
        insModel ： tensor B,num,3

        输出
        "catModel": tensor batch，num，3
        "insModel": tensor batch，num，3
        "deformed": tensor batch，num，3
        "simLoss": chamfer_loss}

        """
        # num*3 的都变成 3*8的，原代码要求
        catModel = catModel.transpose(1, 2)
        catKeys = catKeys.transpose(1, 2)
        insKeys = insKeys.transpose(1, 2)
        insModel = insModel.transpose(1, 2)


        # 根据关键点将类模型变形为实例模型
        B, _, _ = catModel.shape  # 记录batch
        self.insModel = insModel  # 记录实例模型
        source_keypoints = catKeys
        target_keypoints = insKeys
        cage = self.template_vertices
        cage = self.optimize_cage(cage, catModel)
        influence = catInflu

        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints

        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, weights, _ = deform_with_MVC(
            cage, new_cage, self.template_faces.expand(B, -1, -1), catModel.transpose(1, 2), verbose=True)
        
        self.deformed_shapes = deformed_shapes

        # 计算变形的模型和实例模型之间的相似度
        chamfer_loss = pytorch3d.loss.chamfer_distance(
            self.deformed_shapes, rearrange(self.insModel, 'b d n -> b n d'))[0]

        # 输出
        catModel = catModel.transpose(1, 2)
        insModel = insModel.transpose(1, 2)
        catKeys = catKeys.transpose(1, 2)
        insKeys = insKeys.transpose(1, 2)
        outputs = {
            "catModel": catModel,
            "insModel": insModel,
            "catKeys": catKeys,
            "insKeys": insKeys,
            "deformed": self.deformed_shapes,
            "simLoss": chamfer_loss}
        # outputs.update()
        
        return outputs




if __name__ == "__main__":
    from LossCatSim_keyPtsDefor_utils.cages import deform_with_MVC
    from LossCatSim_keyPtsDefor_utils.utils import normalize_to_box, sample_farthest_points
    from utils import SavePoints_asPLY
    # 测试，输入 ： 类模型、类关键点、influence、gt实例关键点inNOCS、实例模型   能不能得到变形的模型和相似度loss
    # 加载数据
    # 读取关键点，类的关键点可能有些有问题，统一一个存起来
    import _pickle as cPickle
    import pickle
    import pprint
    # 读取实例关键点
    data_dir = '/data1/jl/GPVNet/data/NOCS/KeyPoints/camera_train.pkl'
    file = open(data_dir, "rb")
    data = pickle.load(file)
    catKeys = data['1038e4eac0e18dcce02ae6d2a21d494a_source_keypoints']
    insKeys = data['1038e4eac0e18dcce02ae6d2a21d494a_target_keypoints']
    # 读取类关键点和influence
    data_dir = '/data1/jl/GPVNet/data/NOCS/KeyPoints/CateInfluenceMax.pkl'
    file = open(data_dir, "rb")
    data = pickle.load(file)
    catInfluence = data['mug']
    # 读取对应的类模型和实例模型
    # 读取类模型
    mean_shapes = np.load('/data1/jl/6pack/assets/mean_points_emb.npy')
    # cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    catModel = mean_shapes[5]  # mug是最后一个模型
    catModel = mean_shapes[5]  # mug是最后一个模型
    # 读取对应的实例模型
    data_dir = '/data1/jl/awsl-JL/object-deformnet-master/object-deformnet-master/data/obj_models/camera_train.pkl'
    file = open(data_dir, "rb")
    data = pickle.load(file)
    insModel = data['1038e4eac0e18dcce02ae6d2a21d494a']

    # 计算
    catSimLoss = CatSimLoss()
    catModel = torch.tensor(catModel).unsqueeze(0).cuda()
    catKeys = torch.tensor(catKeys).unsqueeze(0).cuda()
    catInfluence = torch.tensor(catInfluence).cuda()
    insKeys = torch.tensor(insKeys).unsqueeze(0).cuda()
    insModel = torch.tensor(insModel).unsqueeze(0).cuda()

    outputs = catSimLoss(catModel, catKeys, catInfluence, insKeys, insModel)

    # 将gt的类模型、实例模型、deformed实例模型保存，确认正确
    # 先将数据变成numpy
    catPoints = outputs["catModel"].cpu().numpy()[0]
    insPoints = outputs["insModel"].cpu().numpy()[0]
    catKeys = outputs["catKeys"].cpu().numpy()[0]
    insKeys = outputs["insKeys"].cpu().numpy()[0]
    deformPoints = outputs["deformed"].cpu().detach().numpy()[0]
    print(catPoints.shape, insPoints.shape, deformPoints.shape)
    SavePoints_asPLY(catPoints, "/data1/jl/6pack/SelfResults/others/testSimCatLoss/catModel.ply")
    SavePoints_asPLY(insPoints, "/data1/jl/6pack/SelfResults/others/testSimCatLoss/insModel.ply")
    SavePoints_asPLY(catKeys, "/data1/jl/6pack/SelfResults/others/testSimCatLoss/catKeys.ply")
    SavePoints_asPLY(insKeys, "/data1/jl/6pack/SelfResults/others/testSimCatLoss/insKeys.ply")
    SavePoints_asPLY(deformPoints, "/data1/jl/6pack/SelfResults/others/testSimCatLoss/deformed.ply")

