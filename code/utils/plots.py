import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util

# 绘制模型预测结果
# 输入：
# model：模型
# indices：图像索引
# model_outputs：合并后的模型输出
# pose：pose输入
# rgb_gt：rgb真值
# path：绘制结果保存路径
# epoch：训练epoch
# img_res：图像分辨率
# 其他：绘制参数
def plot(model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, max_depth, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    # 获取模型输出并reshape，用于后续plot
    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)

    # 计算camera下的深度值(即camera坐标系下的z值)
    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size,-1)

    # 获取相机位置和z轴方向
    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)

    # plot rendered images
    # 绘制渲染图像
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    # 绘制深度图
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

    data = []

    # plot surface
    # 绘制曲面
    surface_traces = get_surface_trace(path=path,
                                       epoch=epoch,
                                       sdf=lambda x: model.implicit_network(x)[:, 0],
                                       resolution=resolution
                                       )
    data.append(surface_traces[0])

    # plot cameras locations
    # 可视化相机位置和方向
    for i, loc, dir in zip(indices, cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    # 可视化射线与表面的交点
    for i, p, m in zip(indices, points, network_object_mask):
        p = p[m]
        sampling_idx = torch.randperm(p.shape[0])[:2048]
        p = p[sampling_idx, :]

        val = model.implicit_network(p)
        caption = ["sdf: {0} ".format(v[0].item()) for v in val]
        
        # 每张图像对应的交点散点图
        data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))

    # 绘图
    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch) # 做成离线的html文件
    offline.plot(fig, filename=filename, auto_open=False)


# 散点图
def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


# 相机pose可视化(位置+方向)
# points: shape [N, 3]，表示箭头起点的 3D 坐标
# directions: shape [N, 3]，表示箭头的方向向量（从起点出发，指向的方向）
# colors: shape [N, 3]，表示箭头颜色(默认为红色: '#bd1540')
def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(), # 位置
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(), # 方向
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute', # 控制箭头大小的方式（absolute 表示不缩放）
        sizeref=0.125,       # 箭头大小缩放参考因子（越小越长）
        showscale=False,     # 是否显示颜色图例（不显示）
        colorscale=[[0, color], [1, color]], # 颜色渐变：统一颜色
        anchor="tail" # 锥体锚定在“尾部”，即从 `point` 出发
    )

    return trace


# 获取表面
def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    # 在单位球上采样100*100*100个3D网格点，用于计算sdf值
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    # 计算sdf值
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)  # 计算得到所有sdf值

    if (not (np.min(z) > 0 or np.max(z) < 0)): # 判断sdf结果是否有效

        z = z.astype(np.float32)
        
        # 提取sdf=0的等值面
        # verts：mesh顶点
        # faces：三角面片，每个face数据是3个vert的索引
        # normals：vert的法向量
        # values：vert的等值面值(接近0)
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),  # 顺序：[y, x, z] -> [x, y, z]
            level=0, # 0等值面
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1], # x坐标的间距
                     grid['xyz'][0][2] - grid['xyz'][0][1], # y坐标的间距
                     grid['xyz'][0][2] - grid['xyz'][0][1]))# z坐标的间距

        # marching_cubes输出的verts是相对于左上角[0, 0, 0]的坐标，所以需要加上偏移转到真实坐标范围[-1， 1]
        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose() # I第一个顶点，J第二个顶点，K第三个顶点

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


# 生成3D grid点坐标[x, y, z]; x,y,z ∈ [-1, 1]
def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution) # [100,]
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)  # xx: [100, 100, 100]
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float) # [1000000, 3]

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    # reshape depth maps
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1), # 单通道扩为三通道
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0) # [C, H, W] -> [H, W, C]
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    # 保存深度图
    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    # 将像素值范围从[-1, 1] 转换为[0, 1]
    ground_true = (ground_true.cuda() + 1.) / 2.
    rgb_points = (rgb_points + 1. ) / 2.

    # 将模型渲染图像和真值图像横向拼接
    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)  # [B, C, H, W]

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0) # [H, W, C]
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8) # 转为u8

    # 保存图像
    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

# reshape图像数据: [batch_size, num_samples, channels] -> [batch_size, channels, H, W]
def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
