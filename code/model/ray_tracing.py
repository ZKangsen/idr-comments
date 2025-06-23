import torch
import torch.nn as nn
from utils import rend_util

class RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0, # 物体的包围球半径
            sdf_threshold=5.0e-5,       # SDF收敛阈值(表示这个点位于表面)
            line_search_step=0.5,       # sdf值<0时，反向线搜索步长
            line_step_iters=1,          # sdf值<0时，反向线搜索迭代次数
            sphere_tracing_iters=10,    # 球面追踪算法迭代次数
            n_steps=100,                # 未收敛到表面的点在[t_min, t_max]区间的采样数量
            n_secant_steps=8,           # 割线算法迭代次数
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def forward(self,
                sdf,           # sdf函数，实际是调用sdf网络，根据输入推理得到sdf值
                cam_loc,       # 相机位置，[batch_size, 3]
                object_mask,   # 物体mask, [batch_size * num_pixels]
                ray_directions # 射线方向, [batch_size, num_pixels, 3]
                ):

        # 射线方向的数据shape
        batch_size, num_pixels, _ = ray_directions.shape # [batch_size, num_pixels, 3]

        # 计算射线R与球面S_theta交点距离和交点mask
        # 输入：相机位置cam_loc，射线方向ray_directions，单位球体半径r
        # 输出：
        # sphere_intersections: 指的是射线与单位球面的交点与cam_loc的距离，即O+td中的t: t_near, t_far
        # mask_intersect: 记录是否存在交点的mask, True：存在2个交点，False: 存在0或1个交点
        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)

        # 球面跟踪算法
        # 输入: 
        # batch_size: 批次大小
        # num_pixels: 像素数量
        # sdf: sdf网络
        # cam_loc: 相机位置, [batch_size, 3]
        # ray_directions: 射线方向, [batch_size, num_pixels, 3]
        # mask_intersect: 射线与单位球面交点mask, [batch_size, num_pixels]
        # sphere_intersections: 射线与单位球面交点距离, [batch_size, num_pixels, 2]
        # 输出:
        # curr_start_points: start点坐标, [batch_size * num_pixels, 3]
        # unfinished_mask_start: 需要进一步处理的mask, [batch_size * num_pixels]
        # acc_start_dis: start点距离, [batch_size * num_pixels]
        # acc_end_dis: end点距离, [batch_size * num_pixels]
        # min_dis: 射线与单位球面交点的最小距离, [batch_size * num_pixels]
        # max_dis: 射线与单位球面交点的最大距离, [batch_size * num_pixels]
        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        # 计算满足条件的mask: start点距离 < end点距离，与unfinished_mask_start表示的条件不同
        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        # 处理不收敛的case: 在[t_min, t_max]中均匀采样N个点，
        # 找到一个sdf值sign变化的区间，在该区间中使用割线算法查找近似交点
        sampler_mask = unfinished_mask_start # [batch_size * num_pixels]
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0: # > 0表示存在需要进一步处理的点
            # 将start_dis作为t_min, end_dis作为t_max, 得到采样区间的两个端点sampler_min_max
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda() # [batch_size, num_pixels, 2]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            # 射线采样算法
            # 输入:
            # sdf: sdf网络
            # cam_loc: 相机位置, [batch_size, 3]
            # object_mask: 物体mask, [batch_size * num_pixels]
            # ray_directions: 射线方向, [batch_size, num_pixels, 3]
            # sampler_min_max: 采样区间的两个端点, [batch_size, num_pixels, 2]
            # sampler_mask: 采样mask, [batch_size * num_pixels]
            # 输出:
            # sampler_pts: 射线与物体表面的交点, [batch_size * num_pixels, 3]
            # sampler_net_obj_mask: 射线与物体表面相交的像素的掩码,True表示有交点，False表示无交点, [batch_size * num_pixels]
            # sampler_dists: 射线与物体表面相交的像素的深度, [batch_size * num_pixels]
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )
            # 将新求出的交点和距离添加到curr_start_points和acc_start_dis中
            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            # 将已经求出交点的mask更新到network_object_mask中
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
              .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        print('----------------------------------------------------------------')

        # 如果非训练模式，则返回结果
        if not self.training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis

        ray_directions = ray_directions.reshape(-1, 3) # [batch_size * num_pixels, 3]
        mask_intersect = mask_intersect.reshape(-1)    # [batch_size * num_pixels]

        # in_mask: 真实是物体（GT mask 为 1）但网络没判为物体，且射线没与表面相交的像素
        # out_mask: 既不是物体，也没打中表面的背景区域
        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        out_mask = ~object_mask & ~sampler_mask

        # mask_left_out:
        # 所有需要特殊处理的射线，包括：
        # 1. 被网络遗漏的物体区域（真实是物体但没判中）
        # 2. 背景区域
        # 3. 同时这些射线没有与单位球相交（即不在采样范围）
        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            # 提取出所有需要处理的射线的相机位置和方向
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            # 对于一元二次方程：t^2 + 2 * t * O^T * d + O^T * O - r^2 = 0，
            # 当方程有0或1个解时，并且对应函数开口向上，当t=-b/2a时，函数有最小值，即对应球面与射线的最小距离，所以t=-O^T * d
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        # mask: 射线与单位球有交点，但未找到与物体表面的交点
        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            # 更新min dis，缩小查找范围
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]
            # 找最小的sdf值，对应的点作为表面最近点
            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)
            
            # 更新start点和距离
            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        # 输出：
        # curr_start_points: 射线与物体表面相交的3D点, [batch_size * num_pixels, 3]
        # network_object_mask: 射线与物体表面相交的mask, [batch_size * num_pixels]
        # acc_start_dis: 射线与物体表面相交的深度, [batch_size * num_pixels]
        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    # 球面追踪算法
    # 根据射线与球面交点(近和远交点)，同时从近和远交点同时向表面行进，即增加/较小t值，增加/减小的量为sdf值
    # 因为sdf值表示点到表面的符号距离，所以t±sdf可以使点向表面移动
    # 输入:
    #   batch_size: 批次大小
    #   num_pixels: 像素数量
    #   sdf: sdf函数，调用的sdf网络
    #   cam_loc: 相机位置, [batch_size, 3]
    #   ray_directions: 射线方向, [batch_size, num_pixels, 3]
    #   mask_intersect: 射线与球面是否有交点, [batch_size, num_pixels]
    #   sphere_intersections: 射线与球面交点距离t, [batch_size, num_pixels, 2]
    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        # 根据相机位置cam_loc，交点距离t，射线方向ray_directions计算射线与球面交点坐标sphere_intersections_points，[batch_size, num_pixels, 2, 3]
        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        # 将交点mask展平并初始化unfinished_mask_start和end，用于后面索引交点位置和距离,并记录是否需要继续查找射线与物体表面的交点
        unfinished_mask_start = mask_intersect.reshape(-1).clone() # [batch_size * num_pixels]
        unfinished_mask_end = mask_intersect.reshape(-1).clone()   # [batch_size * num_pixels]

        # Initialize start current points
        # 使用射线与单位球面的近交点来初始化start点和距离，用于后续步进计算物体表面的交点
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float() # [batch_size * num_pixels, 3]
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start] # 近交点坐标
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float() # [batch_size * num_pixels]
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0] # 近交点距离

        # Initialize end current points
        # 同理, 使用射线与单位球面的远交点来初始化end点和距离，用于后续步进计算物体表面的交点
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float() # [batch_size * num_pixels, 3]
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end] # 远交点坐标
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float() # [batch_size * num_pixels]
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1] # 远交点距离

        # Initizliae min and max depth
        # 初始化最大最小距离，即射线与单位球面交点的最大最小距离，作为后续minimal_sdf_points函数的输入
        min_dis = acc_start_dis.clone() # [batch_size * num_pixels]
        max_dis = acc_end_dis.clone() # [batch_size * num_pixels]

        # Iterate on the rays (from both sides) till finding a surface
        # 接下来就是从两个方向来迭代(方向1: start->end, 方向2: end->start), 直到找到一个物体表面
        iters = 0

        # 使用start点的sdf值初始化next_sdf_start，用于后续while循环中sdf值的迭代更新
        next_sdf_start = torch.zeros_like(acc_start_dis).cuda() # [batch_size * num_pixels]
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start]) 

        # 使用end点的sdf值初始化next_sdf_end，用于后续while循环中sdf值的迭代更新
        next_sdf_end = torch.zeros_like(acc_end_dis).cuda() # [batch_size * num_pixels]
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        # 从射线与单位球面的交点开始双向迭代查找射线与物体表面交点，并更新交点坐标，sdf值和mask
        while True:
            # 1. 更新当前sdf值
            # 1.1 更新start点的sdf值，并将 <= 阈值的sdf设为0( <= 阈值表示这些点是物体表面点)
            # 注：对于sdf<0的值在后续会进行backwards，所以这里将<=阈值的sdf都设为了0
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda() # [batch_size * num_pixels]
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            # 1.2 同理，更新end点的sdf值，并将 <= 阈值的sdf设为0( <= 阈值表示这些点是物体表面点)
            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda() # [batch_size * num_pixels]
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # 2. 更新mask
            # 2.1 unfinished_mask_start与sdf值 > 阈值的mask进行'&'操作，即sdf > 阈值的点未到达物体表面，需要继续迭代
            # True: 射线与单位球面有2个交点 && 当前sdf值 > 阈值 && start点距离 < end点距离，表示需要继续迭代查找表面点
            # False: 射线与单位球面有0或1个交点 || 当前sdf值 <= 阈值 || start点距离 >= end点距离，表示不需要继续迭代
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            # 如果unfinished_mask_start和unfinished_mask_end的值都为0，或者迭代达到最大次数，跳出循环
            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1 # 迭代次数加1

            # 3. 更新距离t，当前start和end点以及sdf值
            # 3.1 使用sdf值分别双向更新start和end距离，start_dis+sdf_start和end_dis-sdf_end分别向表面靠拢
            acc_start_dis = acc_start_dis + curr_sdf_start # [batch_size * num_pixels]
            acc_end_dis = acc_end_dis - curr_sdf_end       # [batch_size * num_pixels]

            # 3.2 根据更新后的距离计算start和end点坐标，points.shape: [batch_size * num_pixels, 3]
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # 3.3 用新的start和end点计算sdf值
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            # 4. 对于sdf < 0的点进行回溯，防止start和end点更新过多(比如到达物体表面内部或者sdf网络训练不足或更新步长过大)
            not_projected_start = next_sdf_start < 0 # 计算sdf < 0的mask
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            # 迭代回溯sdf < 0的点，直到mask全部为0或者迭代达到最大次数
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                # 上面3.1中acc_start_dis在更新后，即acc_start_dis = acc_start_dis + curr_sdf_start
                # 得到的start点计算的sdf < 0，则使用指数衰减方式进行回溯，即acc_start_dis -= (1-回溯步长)/2^iters * sdf_start
                # 指数衰减方式有利于逐渐逼近表面点，减少表面点附近的震荡
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                # 计算start点坐标
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                # 同理，使用指数衰减方式回溯end点
                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                # 计算end点坐标
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                # 计算start点和end点的sdf值
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                # 更新mask，即对sdf < 0的点继续进行回溯
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1 # 迭代次数加1

            # 更新mask: 保留start_dis < end_dis的点(即对应的mask值为True)
            # 就是做了一层过滤，使得最终的mask满足:
            # True: 射线与单位球面有2个交点 && 当前sdf值 > 阈值 && start点距离 < end点距离，说明未到达物体表面，需要进一步处理
            # False: 射线与单位球面有0或1个交点 || 当前sdf值 <= 阈值 || start点距离 >= end点距离，说明已到达物体表面或与表面无交点
            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        # 返回射线与物体表面近交点的坐标, 近交点的mask, 近交点到cam_loc的距离, 远交点到cam_loc的距离, 射线与单位球面的近/远交点距离
        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    # 射线采样算法: 对未收敛到物体表面点的case进行采样处理
    # 输入: 
    # sdf: sdf网络
    # cam_loc: 相机位置, [batch_size, 3]
    # object_mask: 物体mask, [batch_size * num_pixels]
    # ray_directions: 射线方向, [batch_size, num_pixels, 3]
    # sampler_min_max: 采样区间的两个端点, [batch_size, num_pixels, 2]
    # sampler_mask: 采样mask, [batch_size * num_pixels]
    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        # 获取shape： [batch_size, num_pixels]
        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        # 初始化采样点[n_total_pxl, 3]和采样距离[n_total_pxl], 用于保存最终找到的交点
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        # 获取采样区间: [0, 1/(n_steps-1), ..., 1]，shape: [1, 1, n_steps]
        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        # 根据intervals_dist和[t_min, t_max]计算n_steps个采样点距离t: [batch_size, num_pixels, n_steps]
        # t_i = t_min + dist_ratio * (t_max - t_min)
        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        # 根据pts_intervals计算n_steps个采样点坐标: [batch_size, num_pixels, n_steps, 3]
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        # 获取sampler_mask中非0索引(即需要采样的射线)
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        # 获取非0索引对应的采样点坐标和采样点距离
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :] # [n_mask, n_steps, 3]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask] # [n_mask, n_steps]

        # 计算n_steps个采样点对应的sdf值
        sdf_val_all = []
        # 分批计算sdf值
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        # 将分批计算的sdf值拼接后reshape成[batch_size * num_pixels, n_steps]
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        # tmp = sign_arr * [[100, 99, ..., 1]], sign_arr中每个元素的取值范围是{0, 1, -1}, tmp.shape: [batch_size * num_pixels, n_steps]
        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        # 获取tmp中每行最小值的列索引, sampler_pts_ind.shape: [batch_size * num_pixels]
        # 由于points中的点坐标是从近到远排列的，因此，argmin返回的列索引，就是最近的且sdf可能有符号变化的点，即可能是最近的表面交点
        sampler_pts_ind = torch.argmin(tmp, -1)
        # 将每行最小值列索引对应的点坐标和距离分别保存至sampler_pts和sampler_dists中
        # 由于mask_intersect_idx是从sampler_mask获取的非0索引，points是使用sampler_mask过滤得到的点坐标，
        # 所以 mask_intersect_idx.size == points.shape[0]
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :] # [n_mask, 3]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind] # [n_mask]

        # 获取sampler_mask对应的分割mask，分割mask中True表示图像中感兴趣物体的表面点, False为其他点(或背景点)
        true_surface_pts = object_mask[sampler_mask]
        # 这里将sdf_val中最近点且sdf<0的mask提取出来，表示此处有sdf符号变化，或者说射线点穿过表面进入了物体内部，需要进一步求交点
        # 其中sampler_pts_ind索引对应的sdf值不一定都是负值(有可能是tmp中的非负最小值，这种情况下射线和物体表面可能没有交点)
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        # true_surface_pts & net_surface_pts得到的mask中，True表示对应索引的点在物体表面(由分割mask确定)，同时该点处的sdf值<0(符号变化，需要进一步使用割线法寻找交点)
        # p_out_mask是取反的mask，表示对应索引的点不在物体上(分割mask为False)，或者该点处的sdf值>=0(无交点)，则属于P_out点，则直接从sdf_val中找到最小值索引来确定点的位置和距离
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum() # P_out点数量
        if n_p_out > 0: # 存在P_out点
            # 那么从sdf_val的n_steps个采样点中找到sdf最小值索引来确定最终采样点的坐标和距离
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1) # [n_p_out]
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        # 使用sampler_mask初始化sampler_net_obj_mask
        sampler_net_obj_mask = sampler_mask.clone()
        # 将射线与物体表面无交点的mask值置为False
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        # 如果是训练模式，则使用net_surface_pts & true_surface_pts的mask求对应交点，这个mask结合了分割mask的先验，能更好的训练模型，达到更好的效果
        # 否则，则使用net_surface_pts作为mask求对应交点，测试或评估模式可以更多的相信模型的预测表面点，提高可用性
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0: # 存在需要进一步求解的交点
            # Get secant z predictions
            # 这里索引有点头大
            # z_high: 对应sdf<0的最近点，且分割mask属于感兴趣物体表面
            # z_low: 对应sdf<0的最近点的前一个点，这里的z_low < z_high
            # sdf_high: 对应z_high的sdf值
            # sdf_low: 对应z_low的sdf值
            # cam_loc_secant: 对应射线的相机位置
            # ray_directions_secant: 对应射线的方向
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            # 使用secant方法计算交点距离
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            # 计算射线与物体表面交点坐标
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        # 输出:
        # sampler_pts: 射线与物体表面的交点, [batch_size * num_pixels, 3]
        # sampler_net_obj_mask: 射线与物体表面相交的像素的掩码,True表示有交点，False表示无交点, [batch_size * num_pixels]
        # sampler_dists: 射线与物体表面相交的像素的深度, [batch_size * num_pixels]
        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''
        # secant方法原理:
        # 类似于二分查找法，这里假设了SDF沿射线是线性变化的，使用线性插值进行近似:
        # SDF(x) = (1-t)*d0 + t*d1 = 0 
        # -> t = d0 / (d0-d1) 
        # -> x = (1-t)*x0 + t*x1 
        # -> x = x0 - d0 * (x1-x0) / (d1-d0)
        # 所以下面循环中每次线性插值得到z_pred，然后计算p_mid，再计算sdf值，
        # 根据sdf的符号来更新上下边界low/high，最终迭代指定次数结束循环
        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    # 对P_out点求最小sdf值对应的点坐标和距离
    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps # 采样数量
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        # 计算n个均匀采样点的距离steps
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        # 获取mask对应的相机位置和射线方向
        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        # 计算对应的3D点坐标
        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        # 分批计算3D点的sdf值
        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        # 找到每行n个sdf值中的最小值和索引
        min_vals, min_idx = mask_sdf_all.min(-1)
        # 根据min_idx索引找到对应的3D点坐标和距离
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        # 输出找到的最小sdf值对应的3D点坐标和距离
        return min_mask_points, min_mask_dist
