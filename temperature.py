import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTemperatureScaling(nn.Module):
    """
    Enhanced Adaptive Temperature Scaling module that preserves input dimensions
    """
    def __init__(self, 
                 num_classes=4,
                 initial_temp=1.5,
                 momentum=0.1,
                 min_temp=0.1,
                 max_temp=5.0):
        super(AdaptiveTemperatureScaling, self).__init__()
        self.num_classes = num_classes
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.momentum = momentum
        
        # 1. Channel-wise base temperature (保持与输入相同的空间维度)
        self.base_temperature = nn.Parameter(torch.ones(1, num_classes, 1, 1) * initial_temp)
        
        # 2. 空间注意力机制 (保持通道数不变)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=1),  # 修改: 输出通道数与输入相同
            nn.Sigmoid()
        )
        
        # 3. 通道注意力模块 (保持通道数不变)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes, num_classes // 2, 1),
            nn.ReLU(),
            nn.Conv2d(num_classes // 2, num_classes, 1),
            nn.Sigmoid()
        )
        
        # 4. 温度预测网络 (保持维度不变)
        self.temp_predictor = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 1),  # 修改: 移除特征拼接,直接使用原始logits
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 1),
            nn.Sigmoid()
        )
        
        # 5. 移动平均温度
        register_buffer = getattr(self, 'register_buffer', None)
        if callable(register_buffer):
            self.register_buffer('running_temp', torch.ones(1, num_classes, 1, 1) * initial_temp)
    
    def get_confidence_weights(self, logits):
        """计算基于置信度的权重"""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)
        confidence_weights = 1 - (entropy / torch.log(torch.tensor(self.num_classes, device=logits.device)))
        # 扩展维度以匹配输入
        return confidence_weights.expand(-1, self.num_classes, -1, -1)
    
    def forward(self, logits):
        # 1. 计算空间注意力 (B, C, H, W)
        spatial_weights = self.spatial_attention(logits)
        
        # 2. 计算通道注意力 (B, C, 1, 1)
        channel_weights = self.channel_attention(logits)
        
        # 3. 获取置信度权重 (B, C, H, W)
        confidence_weights = self.get_confidence_weights(logits)
        
        # 4. 预测温度调节因子 (B, C, H, W)
        temp_factor = self.temp_predictor(logits)
        temp_factor = 0.5 + temp_factor  # 缩放到 [0.5, 1.5]
        
        # 5. 计算有效温度
        effective_temp = (self.base_temperature * 
                         temp_factor * 
                         channel_weights * 
                         confidence_weights)
        
        # 6. 更新移动平均
        if self.training and hasattr(self, 'running_temp'):
            self.running_temp = (self.running_temp * (1 - self.momentum) + 
                               effective_temp.mean(dim=(0, 2, 3), keepdim=True) * self.momentum)
        
        # 7. 限制温度范围
        effective_temp = effective_temp.clamp(self.min_temp, self.max_temp)
        
        # 8. 应用温度缩放
        scaled_logits = logits / effective_temp
        
        return scaled_logits
    
    def get_temperature(self):
        """返回所有通道的平均温度"""
        return self.base_temperature.mean().item()
    
    def get_channel_temperatures(self):
        """返回每个通道的温度"""
        return self.base_temperature.squeeze()
    
    def reset_running_stats(self):
        """重置移动平均统计"""
        if hasattr(self, 'running_temp'):
            self.running_temp.fill_(self.base_temperature.mean().item())