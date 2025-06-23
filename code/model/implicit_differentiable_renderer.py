import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

# 隐式网络(SDF网络)
class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size, # 特征向量大小: 256
            d_in,  # 输入维度：3，原始3D位置信息
            d_out, # 输出维度：1，SDF值
            dims,  # 隐层维度：[512, 512, 512, 512, 512, 512, 512, 512]
            geometric_init=True, # 是否使用几何初始化
            bias=1.0,   # 偏置初值
            skip_in=(), # 特殊层：引入原始输入信息
            weight_norm=True, # 是否使用权重归一化
            multires=0 # 多分辨率编码器的层数
    ):
        super().__init__()

        # dims = [输入维度，隐层维度(8个512)，输出维度+特征向量维度(257)]
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0: # 使用多分辨率编码器(增加高频信息)
            embed_fn, input_ch = get_embedder(multires) # 嵌入函数，嵌入向量维度
            self.embed_fn = embed_fn
            dims[0] = input_ch # 嵌入向量维度作为输入维度

        self.num_layers = len(dims) # 网络层数
        self.skip_in = skip_in # 特殊层：引入原始输入信息

        # 构建MLP网络
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                # 对于skip层的前一层，输出的维度=隐层维度-原始输入维度
                # 然后skip层的输入就是前一层的输出+原始输入
                out_dim = dims[l + 1] - dims[0]
            else:
                # 当前层的输出维度
                out_dim = dims[l + 1]

            # 线性层(全连接层)
            lin = nn.Linear(dims[l], out_dim)

            # 初始化网络参数, 参考paper: Sal: Sign agnostic learning of shapes from raw data
            if geometric_init:
                if l == self.num_layers - 2:
                    # 对于MLP的最后一层(即SDF网络的输出层)
                    # weight初始化为均值sqrt(pi) / sqrt(当前层输入维度)，标准差为0.0001的高斯分布
                    # bias初始化为conf文件中的-bias(-0.6)
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    # 对于多分辨率编码器的第一层
                    # 将多分辨率编码中对应原始3D位置输入的权重(即weights的0,1,2列)
                    # 初始化为均值0，标准差为sqrt(2) / sqrt(当前层输出维度)的高斯分布，其余权重初始化为0
                    # bias初始化为0
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    # 对于skip层(本层输入包含了原始编码输入)
                    # bias初始化为0
                    # 将多分辨率编码中对应非原始3D位置输入的权重初始化为0，
                    # 其余权重初始化为均值0，标准差为sqrt(2) / sqrt(当前层输出维度)的高斯分布
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    # 对于其他层，bias初始化为0，权重初始化为均值0，标准差为sqrt(2) / sqrt(当前层输出维度)的高斯分布
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                # 归一化权重
                lin = nn.utils.weight_norm(lin)

            # 网络层：lin0, lin1, ...
            setattr(self, "lin" + str(l), lin)

        # 激活函数：softplus
        self.softplus = nn.Softplus(beta=100)

    # 前向传播
    def forward(self, input, compute_grad=False):
        # input: [batch_size, 3]
        # 使用多分辨率编码器对输入进行编码
        # 编码后的输入维度为：[batch_size, 3 + multires*3*2]
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        # 前向传播
        for l in range(0, self.num_layers - 1):
            # 获取当前linear层
            lin = getattr(self, "lin" + str(l))

            # 如果是skip层，将前一层的输出和原始输入进行拼接作为l层的输入
            # 否则，直接使用前一层的输出作为l层的输入
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            # 如果不是最后一层，使用softplus激活函数
            if l < self.num_layers - 2:
                x = self.softplus(x)

        # 网络输出：[batch_size, d_out + feature_vector_size]
        return x

    def gradient(self, x):
        # 启用输入的梯度计算
        x.requires_grad_(True)
        # 前向传播获取SDF值(只取第一个输出通道)
        y = self.forward(x)[:,:1]
        # 创建与输出形状相同的全1张量作为梯度传播的初始值
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # 计算梯度(自动微分)
        gradients = torch.autograd.grad(
            outputs=y,            # 网络输出
            inputs=x,             # 网络输入
            grad_outputs=d_output, # 梯度传播的初始值(全1)
            create_graph=True,    # 保留计算图以支持高阶导数
            retain_graph=True,    # 保留计算图以备后续使用
            only_inputs=True      # 仅计算对输入的梯度
        )[0]

        # 增加一个维度返回 [batch_size, 1, 3]
        # 这个梯度后面会用于计算表面S_theta在点x处的法向量和射线R(t)与表面S_theta的交点
        return gradients.unsqueeze(1)

# 渲染网络(外观网络)
class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size, # 特征向量大小：256
            mode,  # 渲染模式：idr/no_view_dir/no_normal
            d_in,  # 输入维度：9，if mode==idr: [3D位置x，视角方向v，法向量n]
            d_out, # 输出维度：3，[R, G, B]
            dims,  # 隐层维度：[512, 512, 512, 512]
            weight_norm=True, # 是否使用权重归一化
            multires_view=0   # 多分辨率编码器的层数
    ):
        super().__init__()

        # 渲染模式：idr/no_view_dir/no_normal
        self.mode = mode
        # dims = [输入维度+特征向量维度，隐层维度(4个512)，输出维度(3)]
        dims = [d_in + feature_vector_size] + dims + [d_out]

        # 如果使用多分辨率编码器，将原始视角方向v编码后的维度作为输入维度
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        # 网络层数
        self.num_layers = len(dims)

        # 构建MLP网络
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU() # 隐层使用relu
        self.tanh = nn.Tanh() # 输出层使用tanh

    # 前向传播
    def forward(self, points, normals, view_dirs, feature_vectors):
        # 如果使用多分辨率编码器，对视角方向v进行编码
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr': # 全量模型
            # 拼接输入：[3D位置x，视角方向v，法向量n，特征向量]，shape: [batch_size, 3+(3+4*3*2)+3+256] 
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir': # 去掉视角方向v
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal': # 去掉法向量n
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        # 前向传播, 隐层使用relu
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        # 输出层使用tanh
        x = self.tanh(x)
        return x # 输出RGB值，[batch_size, 3]

class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size') # 特征向量大小(256)
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network')) # 隐式网络，对应IDR论文中的SDF网络(几何网络)
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network')) # 渲染网络，对应IDR论文中的RGB网络(外观网络)
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer')) # 射线追踪器
        self.sample_network = SampleNetwork()                         # 采样网络
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere') # 物体的包围球半径(单位球)

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]  # 内参矩阵, [batch_size, 4, 4]
        uv = input["uv"]                  # 输入uv坐标, [batch_size, num_pixels, 2]
        pose = input["pose"]              # 相机pose, [batch_size, 4, 4] or 待优化的pose嵌入向量[batch_size, 7]
        object_mask = input["object_mask"].reshape(-1) # 输入的mask, [batch_size * num_pixels]
        
        # 获取射线方向和相机位置
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # 获取射线方向数据维度
        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval() # 设置为评估模式
        with torch.no_grad(): # 禁用梯度计算
            # 核心算法：射线追踪，获取射线和物体表面的交点及交点到相机的距离
            # 输入:
            # sdf: sdf网络
            # cam_loc: 相机位置, [batch_size, 3]
            # object_mask: 物体mask, [batch_size * num_pixels]
            # ray_directions: 射线方向, [batch_size, num_pixels, 3]
            # 输出:
            # points: 射线和物体表面的交点, [batch_size * num_pixels, 3]
            # network_object_mask: 射线与物体表面相交的mask, [batch_size * num_pixels]
            # dists: 交点到相机的距离, [batch_size * num_pixels]
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train() # 设置为训练模式

        # 重新根据dists和ray_dirs计算points
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        # 计算points的sdf值
        sdf_output = self.implicit_network(points)[:, 0:1] # [batch_size * num_pixels, 1]
        ray_dirs = ray_dirs.reshape(-1, 3)  # [batch_size * num_pixels, 3]

        if self.training: # 训练模式
            # 获取属于感兴趣物体的表面mask，以及mask对应的3D点，距离，射线方向，相机位置，sdf输出，3D点的数量
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]   # 3D点
            surface_dists = dists[surface_mask].unsqueeze(-1) # 距离
            surface_ray_dirs = ray_dirs[surface_mask] # 射线方向
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask] # 相机位置
            surface_output = sdf_output[surface_mask] # sdf网络输出
            N = surface_points.shape[0] # 3D点数量

            # Sample points for the eikonal loss
            # 计算eikonal loss对应的采样点(在单位球内[-1, 1]均匀采样)，数量是batch_size * num_pixels的一半
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            # 将points和eikonal_points拼接起来, 其中points进行detach，防止梯度回传，因为这里只是需要数值作为输入，
            # 然后在implicit_network.gradient中显式调用pytorch的自动求导来计算sdf值相对于3D点的梯度，不需要加入整个网络梯度图
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            # 拼接表面点和eikonal点得到所有点
            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            # 计算表面点的sdf值
            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            # 使用pytorch自动求导来计算sdf值相对于3D点的梯度
            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :] # 用于计算eikonal loss: sdf函数的梯度模长=1

            # 将射线与物体表面交点表示成关于隐式几何和相机参数的可导函数，见paper的公式3
            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            # eval模式下，直接根据network_object_mask提取对应3D点
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        # 将射线方向取反方向：这是由光场模型决定的，根据paper中使用的物体表面光场公式，
        # -v表示场景中光沿着射线方向 w^o = -v 到达相机中心, 所以rgb网络中view_dir的输入是-v
        # rgb网络拟合的实际是物体表面的光场模型(比较复杂)
        view = -ray_dirs[surface_mask]

        # 初始化rgb值
        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            # 通过rgb网络计算rgb值
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

        # 网络输出
        output = {
            'points': points,     # 射线与物体表面交点
            'rgb_values': rgb_values, # 网络输出的射线与物体表面交点的rgb值
            'sdf_output': sdf_output, # 网络输出的射线与物体表面交点的sdf值
            'network_object_mask': network_object_mask, # 网络输出的射线与物体表面交点的mask
            'object_mask': object_mask, # 物体的分割mask
            'grad_theta': grad_theta    # 网络输出的射线与物体表面交点的sdf值对坐标的梯度，即法向量
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        # 根据sdf网络计算sdf值，然后求sdf对3D点的梯度
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]  # 这个梯度就是物体表面的法向量

        # sdf网络输出的特征向量
        feature_vectors = output[:, 1:]
        # 使用rgb网络计算rgb值
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals
