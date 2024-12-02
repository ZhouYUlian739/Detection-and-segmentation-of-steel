import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import *
import torch.nn.functional as F
from loss import CombinedLoss, EnhancedCombinedLoss
import numpy as np
from calc_miou import compute_mIoU
from calc_miou import fast_hist, per_class_iu
from net import self_net 
from temperature import AdaptiveTemperatureScaling
from radam import RAdam
from data1 import *  # 更新导入
torch.cuda.empty_cache()

# Configuration
OPTIMIZER = 'RAdam'
SCHEDULER = 'ReduceLROnPlateau'
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
INITIAL_TEMPERATURE = 1.5
TEMP_LR_FACTOR = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = 'train_image'

# 定义权重保存路径
MODEL_SAVE_DIR = 'params'
TEMP_SAVE_DIR = 'params'

# 创建必要的目录
for dir_path in [save_path, MODEL_SAVE_DIR, TEMP_SAVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def validate_model(model, temperature_module, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_classes = 4
    epoch_hist = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        val_pbar = tqdm.tqdm(test_loader, desc='Validation')
        for image, segment_image, ids in val_pbar:
            image = image.to(device).float()
            segment_image = segment_image.to(device).float()
            
            output = model(image)
            output = temperature_module(output)
            loss = criterion(output, segment_image)
            
            running_loss += loss.item()
            
            # Calculate IoU
            preds = torch.argmax(output, dim=1)
            seg_gts = torch.argmax(segment_image, dim=1)
            batch_hist = fast_hist(seg_gts.cpu().numpy().flatten(), 
                                 preds.cpu().numpy().flatten(), 
                                 num_classes)
            epoch_hist += batch_hist
            
            val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(test_loader)
    mIoUs = per_class_iu(epoch_hist)
    mean_iou = np.nanmean(mIoUs)
    
    return avg_loss, mean_iou, mIoUs

def train_model(num_epochs, train_loader, test_loader, model, temperature_module, optimizer, criterion, device, save_path, scheduler):
    if not os.path.exists(f"npy/{model.__class__.__name__}_train_predictions"):
        os.makedirs(f"npy/{model.__class__.__name__}_train_predictions")
    
    best_val_miou = 0.0

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        num_classes = 4
        epoch_hist = np.zeros((num_classes, num_classes))
        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for i, (image, segment_image, ids) in enumerate(train_pbar):
            image = image.to(device).float()
            segment_image = segment_image.to(device).float()
            
            output = model(image)
            current_temps = temperature_module.get_channel_temperatures()
            output = temperature_module(output)
            loss = criterion(output, segment_image)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temperature_module.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate IoU for this batch
            preds = torch.argmax(output, dim=1)
            seg_gts = torch.argmax(segment_image, dim=1)
            batch_hist = fast_hist(seg_gts.cpu().numpy().flatten(), 
                                 preds.cpu().numpy().flatten(), 
                                 num_classes)
            epoch_hist += batch_hist

            temp_str = ' '.join([f'T{i}:{t:.2f}' for i, t in enumerate(current_temps)])
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Temps': temp_str
            })

            if i % 10 == 0:
                _image = image[0].cpu().detach()
                _segment_image = segment_image[0].cpu().detach()
                _colored_segment_image = apply_color_map(_segment_image)
                preds = (torch.sigmoid(output[0].cpu().detach()) > 0.5).float()
                _colored_out_image = apply_color_map(preds)
                img = torch.stack([_image, _colored_segment_image, _colored_out_image], dim=0)
                save_image(img, f'{save_path}/{epoch}_{i}.png')

        # Calculate training metrics
        train_avg_loss = running_loss / len(train_loader)
        train_mIoUs = per_class_iu(epoch_hist)
        train_mean_iou = np.nanmean(train_mIoUs)

        # # Validation phase
        # print("\nRunning validation...")
        # val_loss, val_mean_iou, val_mIoUs = validate_model(
        #     model, temperature_module, test_loader, criterion, device
        # )

        # Print metrics
        print(f'\nEpoch {epoch}:')
        print(f'Training - Average Loss: {train_avg_loss:.4f}, mIoU: {train_mean_iou:.4f}')
        # print(f'Validation - Average Loss: {val_loss:.4f}, mIoU: {val_mean_iou:.4f}')
        print(f'Current Temperatures: {temp_str}')
        
        print("\nPer-class IoU:")
        # for class_idx, (train_iou, val_iou) in enumerate(zip(train_mIoUs, val_mIoUs)):
        #     print(f'Class {class_idx+1} - Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}')
        
        for class_idx, train_iou in enumerate(train_mIoUs):
            print(f'Class {class_idx+1} - Train IoU: {train_iou:.4f}')
        
        # Save best model based on validation mIoU
        # if val_mean_iou > best_val_miou:
        #     best_val_miou = val_mean_iou
        torch.save(model.state_dict(), f'{MODEL_SAVE_DIR}/best_model_on_A2.pth')
        torch.save(temperature_module.state_dict(), 
                    f'{TEMP_SAVE_DIR}/best_temperature_module.pth')
        print('New best model saved!')

        scheduler.step(running_loss / len(train_loader))  # 使用验证集损失来调整学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')

if __name__ == '__main__':
    num_classes = 4

    train_loader = DataLoader(TrainDataset(path="Data", mode="training"), batch_size=16, shuffle=True)
    val_loader = DataLoader(ValDataset(), batch_size=8, shuffle=True)
    
    net = self_net().to(device)
    temperature_module = AdaptiveTemperatureScaling(
        num_classes=num_classes,
        initial_temp=INITIAL_TEMPERATURE
    ).to(device)

    # 加载模型权重
    model_weight_path = f'{MODEL_SAVE_DIR}/best_model_on_A2.pth'
    if os.path.exists(model_weight_path):
        net.load_state_dict(torch.load(model_weight_path))
        print('Successfully loaded model weights!')
    else:
        print('No model weights found, starting from scratch.')

    # 加载温度模块权重
    temp_weight_path = f'{TEMP_SAVE_DIR}/best_temperature_module.pth'
    if os.path.exists(temp_weight_path):
        temperature_module.load_state_dict(torch.load(temp_weight_path))
        print('Successfully loaded temperature module weights!')
    else:
        print('No temperature module weights found, starting with initial temperature.')

    # 对温度模块使用较小的学习率
    params = [
        {'params': net.parameters()},
        {'params': temperature_module.parameters(), 'lr': 0.01 * TEMP_LR_FACTOR}
    ]
    
    optimizer = RAdam(
        params, 
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=False
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True
    )

    criterion = EnhancedCombinedLoss(num_classes=4)

    num_epochs = 10
    train_model(num_epochs, train_loader, val_loader, net, temperature_module, optimizer, criterion, device, save_path, scheduler)