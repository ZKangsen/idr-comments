import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F

def load_rgb(path):
    img = imageio.imread(path) # 读取图像 [H, W, 3]
    img = skimage.img_as_float32(img) # 图像转为float32并将像素值归一化(pixel/255.0)

    # pixel values between [-1,1]
    img -= 0.5 # 将像素值归一化到[-0.5, 0.5]
    img *= 2.  # 将像素值归一化到[-1, 1]
    img = img.transpose(2, 0, 1) # [3, H, W]
    return img

def load_mask(path):
    alpha = imageio.imread(path, mode='L') # 读取mask, [H, W]
    alpha = skimage.img_as_float32(alpha) # mask图像转为float32
    object_mask = alpha > 0.5 # mask值转为bool类型，即感兴趣物体的mask值为1，其他为0
    return object_mask

def load_K_Rt_from_P(filename, P=None):
    # 如果P为None, 则从文件读取P
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    # 根据P分解出内参和pose
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0] # 内参
    R = out[1] # 旋转矩阵，world系到相机系的旋转
    t = out[2] # 平移向量，相机在world系下的位置

    # 将内参和pose转为numpy数组
    K = K/K[2,2]            # 归一化内参(标准内参的K[2, 2]应该是1)
    intrinsics = np.eye(4)  # 初始化为4x4单位矩阵
    intrinsics[:3, :3] = K  # 将K赋值给前3x3部分

    pose = np.eye(4, dtype=np.float32) # 初始化为4x4单位矩阵
    pose[:3, :3] = R.transpose()       # 旋转矩阵，这里R经过了转置，即pose[:3, :3]是相机到world系的旋转
    pose[:3,3] = (t[:3] / t[3])[:,0]   # 平移向量，这里的t是4x1齐次坐标，需要进归一化

    # intrinsics： 4x4
    # pose: 4x4
    return intrinsics, pose

# 获取相机参数: world系下的ray方向和相机位置
def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: # pose: 四元数+平移向量 #In case of quaternion vector representation
        cam_loc = pose[:, 4:] # 相机在world系下的位置
        R = quat_to_rot(pose[:,:4]) # 四元数转旋转矩阵
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float() # [batch_size, 4, 4]
        p[:, :3, :3] = R # 相机到world系的旋转
        p[:, :3, 3] = cam_loc # 相机在world系下的位置
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3] # 相机在world系下的位置
        p = pose # 相机到world系的变换矩阵，[batch_size, 4, 4]

    # 获取uv的shape
    batch_size, num_samples, _ = uv.shape

    # 获取像素的x,y坐标，加入z可得到齐次坐标
    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1) # 改变内存视图, [batch_size, num_samples]
    y_cam = uv[:, :, 1].view(batch_size, -1) # 改变内存视图, [batch_size, num_samples]
    z_cam = depth.view(batch_size, -1) # 改变内存视图, [batch_size, num_samples]

    # 像素转为相机坐标系下的点(齐次坐标[x,y,z,1]), [batch_size, num_samples, 4]
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    # 调整维度顺序，便于矩阵乘法, [batch_size, 4, num_samples]
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    # 将相机坐标点转到world坐标系, [batch_size, num_samples, 3]
    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :] # 计算ray方向, [batch_size, num_samples, 3]
    ray_dirs = F.normalize(ray_dirs, dim=2) # 归一化ray方向, [batch_size, num_samples, 3]

    # 返回ray方向[batch_size, num_samples, 3]和相机位置[batch_size, 3]
    return ray_dirs, cam_loc

# 获取相机位置和方向
def get_camera_for_plot(pose):
    if pose.shape[1] == 7: # 处理四元数情况 #In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else: #处理4x4矩阵 # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]  # z轴方向
    return cam_loc, cam_dir

# 像素坐标转为相机坐标系下的点
def lift(x, y, z, intrinsics):
    # parse intrinsics
    # 获取内参各分量
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    # K: [[fx, sk, cx],   K_inv: [[1/fx, -sk/(fx*fy), cy*sk/(fx*fy)-cx/fx],
    #     [0,  fy, cy],           [0,           1/fy,              -cy/fy],
    #     [0,   0,  1]]           [0,              0,                   1]]
    # 根据像素和内参计算相机系的x,y坐标
    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

# 四元数转旋转矩阵
def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

# 旋转矩阵转四元数
def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q

# 该函数用于计算射线与球面的交点(两个交点：一个近交点，一个远交点)
# 输入:
#   cam_loc: 相机位置, [batch_size, 3]
#   ray_directions: 射线方向, [batch_size, num_pixels, 3]
#   r: 球体半径, 默认值为1.0
# 输出:
# sphere_intersections: 射线与球面交点, [batch_size, num_pixels, 2]
# mask_intersect: 射线与球面是否有交点,  [batch_size, num_pixels]
def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
    '''
    该函数是将射线方程和球面方程进行联合求解交点的公式实现,具体如下：
    1. 相机位置为O(光心), 射线方向为d(单位向量), 注: 这里的相机位置和射线方向都是在world系下的
    2. 射线方程为: x(t) = O + t * d, t>=0, 注: 这个射线方程是在world系下定义的
    3. 球面方程为: ||x||^2 = r^2, 注: 单位球体的球心在世界坐标系的原点O_w, 半径为r=1
    4. 将射线代入球面方程, 可得：
        ||O+td||^2 = r^2 ==> (O+td)^T * (O+td) = r^2
        ==> O^T * O + 2 * t * O^T * d + t^2 * d^T * d - r^2 = 0
        ==> t^2 * d^T * d + 2 * t * O^T * d + O^T * O - r^2 = 0 
        由于d是单位向量, d^T * d = 1, 所以上式可以简化为：
        t^2 + 2 * t * O^T * d + O^T * O - r^2 = 0
        这是一个关于t的一元二次方程, 可以通过求解这个方程来得到t的值(可能为0, 1或2个实数解), 从而得到射线与球面的交点
    5. 一元二次方程判别式: Δ = b^2 - 4ac (对应下面代码的under_sqrt)
    6. 当Δ>0时, 方程有两个实根, 即射线与球面有两个交点, 分别为t1和t2, 求解如下：
        t1 = (-b + √Δ) / 2a  远点
        t2 = (-b - √Δ) / 2a  近点
        注: 这里的t1和t2是射线方程中的t, 即射线与球面的交点到射线起点的距离
    '''

    # 获取ray方向的shape
    n_imgs, n_pix, _ = ray_directions.shape

    # 计算 ray_cam_dot = O^T * d = ray_directions dot cam_loc
    cam_loc = cam_loc.unsqueeze(-1)  # [batch_size, 3] -> [batch_size, 3, 1]
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze() # [batch_size, num_pixels]
    # 计算判别式 under_sqrt = b^2 - 4ac
    #                     =(2 * O^T * d)^2 - 4 * 1 * (O^T * O - r^2) 
    #                     = 4 * ((O^T * d)^2 - (O^T * O - r^2))
    # 去掉系数4，得到如下 under_sqrt = (O^T * d)^2 - (O^T * O - r^2)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2) # [batch_size, num_pixels]

    under_sqrt = under_sqrt.reshape(-1) # [batch_size, num_pixels] -> [batch_size * num_pixels]
    mask_intersect = under_sqrt > 0 # 计算交点mask, 判别式>0, 即射线与球面有交点, [batch_size * num_pixels]

    # 计算交点
    # 初始化交点, [batch_size * num_pixels, 2]
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    # 计算交点: (-b±√Δ)/2a, 由于a=1, 分母的2已被分子的2约掉，所以交点就是如下实现:
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float() # [batch_size * num_pixels, 2]
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1) # [batch_size * num_pixels, 2]

    # [batch_size * num_pixels, 2] -> [batch_size, num_pixels, 2]
    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2) 
    # 数据截断，将小于0的交点值截断为0, 即射线与球面的交点到射线起点的距离不能为负数
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    # [batch_size * num_pixels] -> [batch_size, num_pixels]
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    # 返回射线与球面的交点和交点mask
    # sphere_intersections: [batch_size, num_pixels, 2]
    # mask_intersect: [batch_size, num_pixels]
    return sphere_intersections, mask_intersect

def get_depth(points, pose):
    ''' Retruns depth from 3D points according to camera pose '''
    # 将points从world系转到camera系，然后将depth(就是z值)提取出来
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    # 转为齐次坐标
    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1) # [batch_size, 4, num_samples]

    points_cam = torch.inverse(pose).bmm(points_hom) # 将world坐标转换到相机坐标系
    depth = points_cam[:, 2, :][:, :, None] # [batch_size, num_samples, 1]
    return depth

