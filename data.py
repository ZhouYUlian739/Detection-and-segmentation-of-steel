import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from utils import *

class BaseDataset(Dataset):
    def __init__(self, path, mode="training", augment=True):
        self.path = path
        self.mode = mode
        self.augment = augment
        self.classes = ["Background", "Inclusions", "Patches", "Scratches"]
        
        self.train_transforms = self.get_train_transforms()
        self.val_transforms = self.get_val_transforms()

    def get_train_transforms(self):
        return A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.RandomBrightnessContrast(p=0.2,brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            # A.Normalize(
            #     mean=[0.442, 0.442, 0.442],
            #     std=[0.126, 0.126, 0.126],
            #     max_pixel_value=255.0
            # ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

    def get_val_transforms(self):
        return A.Compose([
            # A.Normalize(
            #     mean=[0.442, 0.442, 0.442],
            #     std=[0.126, 0.126, 0.126],
            #     max_pixel_value=255.0
            # ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

class TrainDataset(BaseDataset):
    def __init__(self, path="MyData", mode="training", augment=True):
        super().__init__(path, mode, augment)
        # 现在获取指定mode下的所有文件名
        self.name = os.listdir(os.path.join(path, 'Lab'))

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'Lab', segment_name)
        image_path = os.path.join(self.path, 'Img', segment_name.replace('png', 'jpg'))
        
        segment_image = np.array(keep_image_size_open(segment_path))
        image = np.array(keep_image_size_open_rgb(image_path))
        
        if self.mode == 'training' and self.augment:
            transformed = self.train_transforms(image=image, mask=segment_image)
        else:
            transformed = self.val_transforms(image=image, mask=segment_image)
        
        image = transformed['image']
        segment_image = transformed['mask']
        
        mask_one_hot = torch.zeros(len(self.classes), *segment_image.shape[-2:])
        for i in range(len(self.classes)):
            mask_one_hot[i] = (segment_image == i).float()
        
        file_name_without_ext = os.path.splitext(segment_name)[0]
        return image, mask_one_hot, file_name_without_ext

    def __len__(self):
        return len(self.name)

class ValDataset(BaseDataset):
    def __init__(self, path = "DataB", mode="test", augment=True):
        super().__init__(path, mode, augment)
        self.name = os.listdir(os.path.join(path, 'Lab'))
        
        # # 根据mode划分数据
        # if mode == "training":
        #     self.name = self.name[:int(len(self.name) * 0.8)]  # 80%用于训练
        # else:  # testing
        #     self.name = self.name[int(len(self.name) * 0.8):]  # 20%用于测试

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'Lab', segment_name)
        image_path = os.path.join(self.path, 'Img', segment_name.replace('png', 'jpg'))
        
        segment_image = np.array(keep_image_size_open(segment_path))
        image = np.array(keep_image_size_open_rgb(image_path))
        
        if self.mode == 'training' and self.augment:
            transformed = self.train_transforms(image=image, mask=segment_image)
        else:
            transformed = self.val_transforms(image=image, mask=segment_image)
        
        image = transformed['image']
        segment_image = transformed['mask']
        
        mask_one_hot = torch.zeros(len(self.classes), *segment_image.shape[-2:])
        for i in range(len(self.classes)):
            mask_one_hot[i] = (segment_image == i).float()
        
        file_name_without_ext = os.path.splitext(segment_name)[0]
        return image, mask_one_hot, file_name_without_ext

    def __len__(self):
        return len(self.name)

class TestDataset(BaseDataset):
    def __init__(self, path="DataC", mode="training", augment=True):
        super().__init__(path, mode, augment)
        self.img_path = os.path.join(path, 'Img')
        self.name = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]

    def __getitem__(self, index):
        img_name = self.name[index]
        image_path = os.path.join(self.img_path, img_name)
        image = np.array(keep_image_size_open_rgb(image_path))
        transformed = self.val_transforms(image=image)
        image = transformed['image']        
        file_name_without_ext = os.path.splitext(img_name)[0]
        return image, file_name_without_ext

    def __len__(self):
        return len(self.name)


if __name__ == "__main__":
    data = TrainDataset()
    print(data[0][0].unique())
    print(data[0][1].shape)
    print(data[0][1][1].unique())  # 查看分割图像中的唯一值
    _image = data[122][0]
    _segment_image = data[122][1]
    _colored_segment_image = apply_color_map(_segment_image)
    img = torch.stack([_image, _colored_segment_image], dim=0)
    save_image(img, 'A.png')
