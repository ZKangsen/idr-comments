import torch
from torch import nn
from torch.nn import functional as F

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight  # eikonal权重 λ
        self.mask_weight = mask_weight        # mask权重 ρ
        self.alpha = alpha                    # mask近似指示函数中的参数α，α->∞时，近似指示函数接近真正的分段指示函数 
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        # network_object_mask & object_mask表示网络推理mask和分割mask中都存在物体表面
        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0]) # l1 loss
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean() # 将sdf梯度的模应该等于1作为loss
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        # 这里是计算的P_out点对应的mask loss
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0]) # BCE loss
        return mask_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()  # rgb真值
        network_object_mask = model_outputs['network_object_mask'] # 网络预测的物体mask
        object_mask = model_outputs['object_mask'] # 物体mask真值

        # 各种loss
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        # 总loss
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
        }
