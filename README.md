2024年人工智能算法大赛赛题二钢材检测思路
  模型使用：FDDWNET，仅0.8M大小，符合比赛模型参数满分要求
  优化器部分，我们原先使用 Adaw 优化器，后来使用了改进版的 RAdaw 优化器，对初始学习率是具有鲁棒性的，可以适应更宽范围内的变化，拥有更好的训练效果。在需要注意的创新点中，我们在训练过程中引入了温度缩放的策略，使用组合型的
EnhancedCombinedLoss 的损失函数,学习策略用的 ReduceLROnPlateau 进行动态学习率调整。
  
