from PIL import Image
import torch

def keep_image_size_open(path, size=(200, 200)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

def keep_image_size_open_rgb(path, size=(200, 200)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask

# utils.py

import torch

# 定义颜色映射
colors = {
    0: [0, 0, 0],      # 背景 - 黑色
    1: [255, 0, 0],    # 类别1 - 红色
    2: [0, 255, 0],    # 类别2 - 绿色
    3: [0, 0, 255],    # 类别3 - 蓝色
    # 如果有更多类别，请在此添加
}

def apply_color_map(segmentation, num_classes=4):
    """
    将分割图像转换为彩色图像。

    :param segmentation: 形状为 (C, H, W) 的张量，one-hot 编码的分割标签。
    :param num_classes: 类别数，默认 4 个类别。
    :return: 形状为 (3, H, W) 的张量，表示彩色图像。
    """
    C, H, W = segmentation.shape
    # 创建彩色图像的张量 (3, H, W)，每个通道分别对应 RGB
    colored_image = torch.zeros((3, H, W), dtype=torch.uint8)

    # 遍历每个类别通道，将对应的1值设置为颜色
    for class_idx in range(num_classes):
        # 获取该类别的颜色 (R, G, B)
        color = colors.get(class_idx, [255, 255, 255])  # 默认白色
        # 获取该类别的掩码
        mask = segmentation[class_idx] == 1
        # 给 R、G、B 通道赋值
        for i in range(3):
            colored_image[i][mask] = color[i]

    return colored_image
