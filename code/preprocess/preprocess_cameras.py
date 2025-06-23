import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import os

import utils.general as utils

# 读取相机投影矩阵
def get_Ps(cameras,number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64) # 3x4投影矩阵
        Ps.append(P)
    return np.array(Ps)


#Gets the fundamental matrix that transforms points from the image of camera 2, to a line in the image of
#camera 1
# 计算基础矩阵F12, 将像素点从相机2图像转换到相机1图像
def get_fundamental_matrix(P_1,P_2):
    P_2_center=np.linalg.svd(P_2)[-1][-1, :]
    epipole=P_1@P_2_center        # epipole: camera2相机中心在camera1图像中的投影
    epipole_cross=np.zeros((3,3)) # epipole_cross: epipole的反对称矩阵
    epipole_cross[0,1]=-epipole[2]
    epipole_cross[1, 0] = epipole[2]

    epipole_cross[0,2]=epipole[1]
    epipole_cross[2, 0] = -epipole[1]

    epipole_cross[1, 2] = -epipole[0]
    epipole_cross[2, 1] = epipole[0]

    F = epipole_cross@P_1 @ np.linalg.pinv(P_2) # 基础矩阵
    return F



# Given a point (curx,cury) in image 0, get the  maximum and minimum
# possible depth of the point, considering the second image silhouette (index j)
# 计算image0中给定像素点可能的最大最小深度，并考虑image_j中的mask轮廓
# 输入：
# curx,cury: image 0中的像素坐标
# P_j：image_j的投影矩阵
# silhouette_j: image_j的mask轮廓, points: [x,y,1]
# P_0: image 0的投影矩阵
# Fj0: image0中的像素点投影到image_j中的基础矩阵
# j: image_j的索引
def get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j):
    # transfer point to line using the fundamental matrix:
    cur_l_1=Fj0 @ np.array([curx,cury,1.0]).astype(np.float) # 计算image_j中的极线
    cur_l_1 = cur_l_1 / np.linalg.norm(cur_l_1[:2]) # 归一化，为了计算点到直线距离(在点到直线距离计算公式中，会进行归一化)

    # Distances of the silhouette points from the epipolar line:
    dists = np.abs(silhouette_j.T @ cur_l_1) # 计算点到直线距离
    relevant_matching_points_1 = silhouette_j[:, dists < 0.7] # 选择dist<0.7的像素点作为匹配点

    if relevant_matching_points_1.shape[1]==0:
        return (0.0,0.0)
    # 根据image_0和image_j中的像素匹配点进行三角化计算3D点坐标,该3D点位于world系下
    X = cv2.triangulatePoints(P_0, P_j, np.tile(np.array([curx, cury]).astype(np.float),
                                                (relevant_matching_points_1.shape[1], 1)).T,
                              relevant_matching_points_1[:2, :])
    depths = P_0[2] @ (X / X[3]) # 计算3D点在相机0系下的深度
    reldepth=depths >= 0  
    depths=depths[reldepth] # 将>=0的深度提取出来
    if depths.shape[0] == 0:
        return (0.0, 0.0)

    min_depth=depths.min() # 最小深度
    max_depth = depths.max() # 最大深度

    return min_depth,max_depth

#get all fundamental matrices that trasform points from camera 0 to lines in Ps
# 计算所有基础矩阵：从P0到Ps的映射
def get_fundamental_matrices(P_0, Ps):
    Fs=[]
    for i in range(0,Ps.shape[0]):
        F_i0 = get_fundamental_matrix(Ps[i],P_0)
        Fs.append(F_i0)
    return np.array(Fs)

# 获取感兴趣物体的mask坐标点和mask二进制图像
def get_all_mask_points(masks_dir):
    mask_paths = sorted(utils.glob_imgs(masks_dir))
    mask_points_all=[]
    mask_ims = []
    for path in mask_paths:
        img = mpimg.imread(path) # 读取图像，像素值范围:[0, 1]
        cur_mask = img.max(axis=2) > 0.5
        mask_points = np.where(img.max(axis=2) > 0.5)
        xs = mask_points[1] # x坐标
        ys = mask_points[0] # y坐标
        mask_points_all.append(np.stack((xs,ys,np.ones_like(xs))).astype(np.float)) # 感兴趣物体的mask坐标点[x, y, 1]
        mask_ims.append(cur_mask) # mask图像(二进制图像)
    return mask_points_all,np.array(mask_ims)

# 根据初始center和scale重新采样3D点，然后投影至图像系，判断是否在图像中来挑选最终的points，重新计算center和scale
def refine_visual_hull(masks, Ps, scale, center):
    num_cam=masks.shape[0]
    GRID_SIZE=100
    MINIMAL_VIEWS=45 # Fitted for DTU, might need to change for different data.
    im_height=masks.shape[1]
    im_width = masks.shape[2]
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE*GRID_SIZE*GRID_SIZE, 1))
    for i in range(num_cam):
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE*GRID_SIZE*GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.long)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]
        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid=final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid,np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3,final_points.T

# the normaliztion script needs a set of 2D object masks and camera projection matrices (P_i=K_i[R_i |t_i] where [R_i |t_i] is world to camera transformation)
# 计算归一化矩阵
# 输入:
# Ps: 相机投影矩阵
# mask_points_all: 感兴趣物体的mask坐标点[x, y, 1]
# number_of_normalization_points: 样本点数量
# num_cameras: 相机数量
# masks_all: 感兴趣物体的mask二进制图像
def get_normalization_function(Ps,mask_points_all,number_of_normalization_points,number_of_cameras,masks_all):
    P_0 = Ps[0]
    Fs = get_fundamental_matrices(P_0, Ps) # 计算基础矩阵，将P_0投影到Ps中
    P_0_center = np.linalg.svd(P_0)[-1][-1, :]
    P_0_center = P_0_center / P_0_center[3] # 计算相机0的中心点，world系下的坐标

    # Use image 0 as a references
    xs = mask_points_all[0][0, :] # 图像0的x坐标
    ys = mask_points_all[0][1, :] # 图像0的y坐标

    counter = 0
    all_Xs = [] # 所有3D点

    # sample a subset of 2D points from camera 0
    # 随机采样number_of_normalization_points个2D点
    indss = np.random.permutation(xs.shape[0])[:number_of_normalization_points]

    for i in indss:
        curx = xs[i]
        cury = ys[i]
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        # 检查每个3D点的最小/最大深度是否在其他相机中也有效
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10
        for j in range(1, number_of_cameras, 5):
            min_d, max_d = get_min_max_d(curx, cury, Ps[j], mask_points_all[j], P_0, Fs[j], j)

            # 如果最小深度小于0.00001(最小深度接近0)，则表示该点无效
            if abs(min_d) < 0.00001:
                observerved_in_all = False
                break
            max_d_all = np.min(np.array([max_d_all, max_d]))
            min_d_all = np.max(np.array([min_d_all, min_d]))
            # 如果最大深度小于最小深度+0.01(表示最大深度不够大)，则表示该点无效
            if max_d_all < min_d_all + 1e-2:
                observerved_in_all = False
                break
        if observerved_in_all:
            direction = np.linalg.inv(P_0[:3, :3]) @ np.array([curx, cury, 1.0]) # 计算方向
            all_Xs.append(P_0_center[:3] + direction * min_d_all) # 计算最近点
            all_Xs.append(P_0_center[:3] + direction * max_d_all) # 计算最远点
            counter = counter + 1

    print("Number of points:%d" % counter)
    centroid = np.array(all_Xs).mean(axis=0)  # 计算所有点的中心
    # mean_norm=np.linalg.norm(np.array(allXs)-centroid,axis=1).mean()
    scale = np.array(all_Xs).std() # 计算所有点的标准差

    # OPTIONAL: refine the visual hull
    # 将centroid和scale作为初始值生成一些grid点来refine最终的中心点和标准差
    centroid,scale,all_Xs = refine_visual_hull(masks_all, Ps, scale, centroid)

    # 将中心点和标准差放入4x4矩阵中作为scale_mat
    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    normalization[0, 0] = scale
    normalization[1, 1] = scale
    normalization[2, 2] = scale
    return normalization,all_Xs


def get_normalization(source_dir, use_linear_init=False):
    print('Preprocessing', source_dir)

    # 不管是fixed_camera还是trained_camera，都需要计算scale_mat
    if use_linear_init:
        #Since there is noise in the cameras, some of them will not apear in all the cameras, so we need more points
        # 由于相机投影矩阵是带噪声的，所以需要更多点来保证覆盖所有相机视角
        number_of_normalization_points=1000  # 归一化所需点数
        cameras_filename = "cameras_linear_init" # 带噪声相机pose文件名
    else:
        # 相机投影矩阵精度较高，不需要那么多点
        number_of_normalization_points = 100 
        cameras_filename = "cameras" # 相机pose文件名

    masks_dir='{0}/mask'.format(source_dir) # mask图像所在文件夹
    cameras=np.load('{0}/{1}.npz'.format(source_dir, cameras_filename)) # 加载相机pose数据

    # 获取所有mask点
    # mask_points_all: 感兴趣物体的mask坐标
    # mask_all: mask二进制图像
    mask_points_all,masks_all=get_all_mask_points(masks_dir)
    number_of_cameras = len(masks_all) # 相机数量(图像数量，pose数量)
    Ps = get_Ps(cameras, number_of_cameras)  # 所有投影矩阵

    normalization,all_Xs=get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras,masks_all)

    # 保存计算的scale_mat
    cameras_new={}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d'%i]=normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i],np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)

    np.savez('{0}/{1}_new.npz'.format(source_dir, cameras_filename), **cameras_new)

    print(normalization)
    print('--------------------------------------------------------')

    # debug用的，将3D点投影值图像中查看
    if False: #for debugging
        for i in range(number_of_cameras):
            plt.figure()

            plt.imshow(mpimg.imread('%s/%03d.png' % (masks_path, i)))
            xy = (Ps[i,:2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                        Ps[i,2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))

            plt.plot(xy[0, :], xy[1, :], '*')
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='', help='data source folder for preprocess') # 自定义数据路径
    parser.add_argument('--dtu', default=False, action="store_true", help='If set, apply preprocess to all DTU scenes.') # 处理dtu数据
    parser.add_argument('--use_linear_init', default=False, action="store_true", help='If set, preprocess for linear init cameras.') # 带噪声初始化pose

    opt = parser.parse_args()

    if opt.dtu: # 针对dtu数据处理
        source_dir = '../data/DTU'
        scene_dirs = sorted(glob(os.path.join(source_dir, "scan*")))
        for scene_dir in scene_dirs:
            get_normalization(scene_dir,opt.use_linear_init)
    else: # 针对自定义数据处理，需要提供image, mask 和 cameras.npz
        get_normalization(opt.source_dir, opt.use_linear_init)

    print('Done!')
