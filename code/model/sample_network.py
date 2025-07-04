import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    See equation 3 in the paper for more details.
    '''

    # 将射线与物体表面交点表示成关于隐式几何和相机参数的可导函数，见paper的公式3
    # 下面代码是该公式实现: x(θ, τ) = c + t_0 * v - v / (grad(f(x_0;θ_0)) * v_0) * f(c+t_0 * v;θ)
    def forward(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        # t -> t(theta)
        surface_ray_dirs_0 = surface_ray_dirs.detach()
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # t(theta) -> x(theta,c,v)
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v
