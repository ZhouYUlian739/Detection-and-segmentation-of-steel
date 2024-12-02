import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):

        output = torch.sigmoid(output)

        intersection = (output * target).sum(dim=(0, 2, 3))
        union = output.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff
        
        return dice_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        output = torch.sigmoid(output)  # (N, 3, H, W)
        target = target.float()

        BCE_loss = F.binary_cross_entropy(output, target, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, output, target):
        # Ignore background class (index 0)
        dice = self.dice_loss(output[:, 1:], target[:, 1:])
        focal = self.focal_loss(output[:, 1:], target[:, 1:])
        return self.alpha * dice + (1 - self.alpha) * focal

def calculate_iou(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()
    
    assert pred.shape == target.shape, "Shapes of pred and target must match"
    
    # Ignore background class (index 0)
    if len(pred.shape) == 4:
        intersection = (pred[:, 1:] * target[:, 1:]).sum(dim=(2, 3))
        union = pred[:, 1:].sum(dim=(2, 3)) + target[:, 1:].sum(dim=(2, 3)) - intersection
    elif len(pred.shape) == 3:
        intersection = (pred[:, 1:] * target[:, 1:]).sum(dim=(1, 2))
        union = pred[:, 1:].sum(dim=(1, 2)) + target[:, 1:].sum(dim=(1, 2)) - intersection
    else:
        raise ValueError("Unsupported tensor shape for IoU calculation")
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean(dim=1)  # Return mean IoU across the three defect classes



class EnhancedCombinedLoss(nn.Module):
    def __init__(self, num_classes=4, alpha=0.5, beta=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # dice和focal的权重比
        self.beta = beta    # 边界loss的权重
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def get_boundary_mask(self, target, kernel_size=3):
        """
        计算除背景类外的边界mask
        target: (B, C, H, W) 其中已经去除了背景类
        """
        # target已经不包含背景类，直接处理
        target_hard = target > 0.5  # 转换为硬标签
        
        # 定义膨胀和腐蚀的卷积核
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(target.device)
        
        boundary_masks = []
        # 对每个非背景类别计算边界
        for i in range(target.size(1)):  # 遍历每个非背景类别
            cls_target = target_hard[:, i:i+1]
            
            # 膨胀操作
            dilated = F.conv2d(
                cls_target.float(),
                kernel,
                padding=kernel_size//2
            ) > 0
            
            # 腐蚀操作
            eroded = F.conv2d(
                cls_target.float(),
                kernel,
                padding=kernel_size//2
            ) >= kernel_size * kernel_size
            
            # 获取边界
            boundary = (dilated ^ eroded).float()
            boundary_masks.append(boundary)
            
        # 合并所有非背景类的边界 (B, C-1, H, W)
        boundary_mask = torch.cat(boundary_masks, dim=1)
        return boundary_mask
    
    def forward(self, output, target):
        """
        output: 模型预测 (B, C, H, W)
        target: 真实标签 (B, C, H, W)
        都已经移除了背景类
        """
        # 1. 基础的Dice和Focal Loss (已经忽略背景类)
        dice_loss = self.dice_loss(output, target)
        focal_loss = self.focal_loss(output, target)
        
        # 2. 计算边界区域的loss
        boundary_mask = self.get_boundary_mask(target)
        
        # 在边界区域计算BCE loss
        boundary_loss = F.binary_cross_entropy_with_logits(
            output * boundary_mask,  # 只关注边界区域的预测
            target * boundary_mask,  # 边界区域的真实标签
            reduction='mean'
        )
        
        # 3. 组合所有loss
        total_loss = (
            self.alpha * dice_loss + 
            (1 - self.alpha) * focal_loss + 
            self.beta * boundary_loss
        )
        
        return total_loss

# 使用示例
criterion = EnhancedCombinedLoss(num_classes=4)  # 虽然是4类，但实际只处理3类（不含背景）