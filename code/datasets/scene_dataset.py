import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 scan_id=0,
                 cam_file=None
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id)) # 数据路径

        self.total_pixels = img_res[0] * img_res[1] # 图像总像素数量
        self.img_res = img_res # 图像分辨率

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None # 采样索引
        self.train_cameras = train_cameras # 是否优化相机位置

        image_dir = '{0}/image'.format(self.instance_dir) # 图像路径
        image_paths = sorted(utils.glob_imgs(image_dir)) # 图像文件列表
        mask_dir = '{0}/mask'.format(self.instance_dir) # mask路径
        mask_paths = sorted(utils.glob_imgs(mask_dir)) # mask文件列表

        self.n_images = len(image_paths) # 图像数量

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir) # 相机位姿文件
        if cam_file is not None: # 使用指定的相机位姿文件
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        # scale_mat: 相机归一化矩阵，可以理解为将单位球的点转为正常的world系下
        # scale_mat_inv: 是将world系下的坐标转为单位球中的点
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # 相机投影矩阵: K[R,t], world to image
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)] 

        self.intrinsics_all = [] # 内参矩阵4x4
        self.pose_all = [] # 相机pose4x4
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat # 归一化球面系的点转到image的投影矩阵
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P) # 分解P得到的新内参和pose
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float()) # 放入list
            self.pose_all.append(torch.from_numpy(pose).float())  # 放入list
        
        # 读取所有图像数据(像素值范围：[-1, 1])
        self.rgb_images = [] # [n, H*W, 3]
        for path in image_paths:
            rgb = rend_util.load_rgb(path) # [3, H, W]
            rgb = rgb.reshape(3, -1).transpose(1, 0) # [H*W, 3]
            self.rgb_images.append(torch.from_numpy(rgb).float())

        # 读取所有mask数据(像素值：{0, 1})
        self.object_masks = [] # [n, H*W]
        for path in mask_paths:
            object_mask = rend_util.load_mask(path) # [H, W]
            object_mask = object_mask.reshape(-1) # [H*W,]
            self.object_masks.append(torch.from_numpy(object_mask).bool()) # 放入list

    def __len__(self):
        return self.n_images # 图像数量

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)# 获取2D网格坐标矩阵uv，shape: [2, H, W]
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float() # 交换x,y轴，即[y,x]->[x,y]，shape: [2, H, W]
        uv = uv.reshape(2, -1).transpose(1, 0) # shape: [H*W, 2]

        # 采样
        sample = {
            "object_mask": self.object_masks[idx],  # mask, [H*W,]
            "uv": uv,                               # 图像2D坐标索引，[H*W, 2]
            "intrinsics": self.intrinsics_all[idx], # 内参, [4, 4]
        }
        # 真值
        ground_truth = {
            "rgb": self.rgb_images[idx] # 图像，rgb真值, [H*W, 3]
        }

        # 根据ID采样
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        # 不训练相机pose时，使用fixed pose
        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]
        
        # 返回图像ID，采样，真值(rgb)
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    # 获取采样索引
    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size] # 随机采样

    # 获取scale_mat_0
    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    # 获取gt pose(未normalize到单位球面的pose)
    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    # 获取初始pose(待优化的coarse pose)
    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
