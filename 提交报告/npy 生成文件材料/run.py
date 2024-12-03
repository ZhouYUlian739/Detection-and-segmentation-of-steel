import os
import time
from PIL import Image
import tqdm
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.abspath("."))
from model import self_net 

def keep_image_size_open_rgb(path, size=(200, 200)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=[0.442, 0.442, 0.442],
                std=[0.126, 0.126, 0.126],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

    def get_val_transforms(self):
        return A.Compose([
            A.Normalize(
                mean=[0.442, 0.442, 0.442],
                std=[0.126, 0.126, 0.126],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

class DataCDataset(BaseDataset):
    def __init__(self, path="./npy 生成文件材料/DataC", mode="training", augment=True):
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = self_net().to(device)  # Change to FDDWNet
weight_path = f'model.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path, map_location=device))
    print('Successfully loaded weights!')
else:
    raise FileNotFoundError('Model weights not found!')

test_loader = DataLoader(DataCDataset(), batch_size=8, shuffle=False)

def run(test_loader, model, device):
    if not os.path.exists(f"c_test_predictions"):
        os.makedirs(f"c_test_predictions")
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        val_pbar = tqdm.tqdm(test_loader, desc=f'[Test]')
        for i, (image, ids) in enumerate(val_pbar):
            image = image.to(device)
            
            output = model(image)      
            # 使用 argmax 获取每个像素的类别标签
            preds = torch.argmax(output, dim=1)  # 形状为 (N, H, W)

            val_pbar.set_postfix({'Loss': f'--'})

            # 保存预测结果为 .npy 文件
            for pred, id in zip(preds, ids):
                output_np = pred.cpu().numpy()
                np.save(f"c_test_predictions/c_prediction_{id}.npy", output_np)

    elapsed_time = time.time() - start_time
    fps = len(test_loader.dataset) / elapsed_time

    print(f"FPS: {fps:.2f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

run(test_loader, net, device)
