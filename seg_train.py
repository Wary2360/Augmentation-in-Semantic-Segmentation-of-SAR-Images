import os
import numpy as np
import cv2
import warnings
import ssl
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import math
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch.utils as utils
import segmentation_models_pytorch.losses as losses
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from segmentation_models_pytorch.utils.metrics import IoU
import sys
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
import albumentations as albu
import albumentations.augmentations.transforms as AAT
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import wandb
from multiprocessing import freeze_support

if __name__ == '__main__':
    # freeze_support()

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='Train and validate a model with directory paths.')
    # parser.add_argument('--x_train_dir', type=str, default='',
    #                     help='Path to the directory containing training images.')
    # parser.add_argument('--y_train_dir', type=str, default='',
    #                     help='Path to the directory containing training masks.')
    # parser.add_argument('--x_valid_dir', type=str, default='',
    #                     help='Path to the directory containing validation images.')
    # parser.add_argument('--y_valid_dir', type=str, default='',
    #                     help='Path to the directory containing validation masks.')
    # parser.add_argument('--batch_size', type=int, default = 8,
    #                     help='batch size')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='train epochs')
    # parser.add_argument('--project', type=str, default='SAR_segmentation',
    #                     help='wandb project name')
    parser.add_argument('--dataset', type=str, default='',
                        help='dataset_name')
    
    args = parser.parse_args()
    
    # x_train_dir = args.x_train_dir
    # y_train_dir = args.y_train_dir
    # x_valid_dir = args.x_valid_dir
    # y_valid_dir = args.y_valid_dir
    
    # batch_size = args.batch_size
    # epochs = args.epochs``
    
    # project = args.project
    dataset = args.dataset
    
    #▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#
    
    num_workers = 0
    
    x_train_dir = r'dataset/x_train'
    y_train_dir = r'dataset/y_train'
    x_valid_dir = r'dataset/x_valid'
    y_valid_dir = r'dataset/y_valid'
    
    batch_size = 32
    epochs = 60
    
    learning_rate = 32e-4
    weight_decay = 0
    
    seed = 42
    
    project = 'SAR_segmentation_sn6aug' # wandb project name
    # dataset = 'spacenet6' # dataset name
    
    #▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲#

    def seed_everything(seed=42):
        """모든 난수 생성기의 시드를 고정합니다."""
        random.seed(seed)  # Python 내장 random 모듈
        os.environ["PYTHONHASHSEED"] = str(seed)  # Python 해시 생성에 사용되는 시드
        np.random.seed(seed)  # NumPy의 난수 생성기
        torch.manual_seed(seed)  # CPU 연산을 위한 PyTorch 시드
        torch.cuda.manual_seed(seed)  # GPU 연산을 위한 PyTorch 시드
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경을 위한 PyTorch 시드
        torch.backends.cudnn.deterministic = True  # CUDA의 결정론적 알고리즘 사용 설정
        torch.backends.cudnn.benchmark = False  # 네트워크 구조가 변하지 않을 때 성능 향상을 위해 True 설정 가능

    # 시드 고정 함수 호출
    seed_everything(seed=42)
    
    name = f'{dataset}_{batch_size}_{epochs}'
    wandb.init(project=project, name=name, config=vars(args))
    
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)): 
        ssl._create_default_https_context = ssl._create_unverified_context

    workspace_path = './'
    segmentation_path = os.path.join(workspace_path, 'segmentation_models')

    class Dataset(BaseDataset):
        
        CLASSES = ['building']
        
        def __init__(
                self, 
                images_dir, 
                masks_dir, 
                classes=None, 
                augmentation=None, 
                preprocessing=None,
        ):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
            
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

            for i in range(len(self.masks_fps)):
                self.mask_ids = np.unique(cv2.imread(self.masks_fps[i], 0))[1:]
                if len(self.mask_ids) == len(self.class_values):
                    break
            
            self.augmentation = augmentation
            self.preprocessing = preprocessing
        
        def __getitem__(self, i):
            
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i])
            
            mask = np.any(mask > 0, axis=-1).astype('float')
            mask = np.expand_dims(mask, axis=-1)

            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
            return image, mask
            
        def __len__(self):
            return len(self.ids)

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    
#▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼#

    def get_preprocessing(preprocessing_fn):
        
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return albu.Compose(_transform)

    def get_training_augmentation(datset):
        if dataset == "baseline":
            train_transform = [
                
            ]
            
        elif dataset == "Pad Resize":
            train_transform = [
                
            ]
            
        elif dataset == "Distorted Resize":
            train_transform = [
                
            ]
            
        elif dataset == "Random Crop":
            train_transform = [
                
            ]
            
        elif dataset == "Random Crop and Resize":
            train_transform = [
                
            ]
            
        elif dataset == "Horizontal Flip":
            train_transform = [
                albu.HorizontalFlip(p=0.5)
            ]
            
        elif dataset == "Vertical Flip":
            train_transform = [
                albu.VerticalFlip(p=0.5)
            ]
            
        elif dataset == "Rotation90":
            train_transform = [
                albu.RandomRotate90(p=1.0)
            ]
            
        elif dataset == "Fine Rotation [-10, 10]":
            train_transform = [
                albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10, p=0.5, border_mode=0)
            ]
            
        elif dataset == "ShearX [-10, 10]":
            train_transform = [
                albu.Affine(shear={'x': (-10, 10)}, p=0.5)
            ]
            
        elif dataset == "ShearY [-10, 10]":
            train_transform = [
                albu.Affine(shear={'y': (-10, 10)}, p=0.5)
            ]
            
        elif dataset == "Random Erasing":
            train_transform = [
                albu.augmentations.CoarseDropout(max_holes=6, max_height=40,
                    max_width=40, min_holes=2, min_height=30, min_width=30,
                    fill_value=0.0, mask_fill_value=0)
            ]
            
        elif dataset == "Motion Blur":
            train_transform = [
                albu.MotionBlur(blur_limit=11)
            ]
            
        elif dataset == "Sharpening":
            train_transform = [
                AAT.Sharpen(alpha=(0.1,0.4), lightness=(.9,1.0))
            ]
            
        elif dataset == "CLAHE":
            train_transform = [
                albu.Sequential([
                    AAT.FromFloat(dtype='uint8', max_value=255),
                    AAT.CLAHE(clip_limit=4.0),
                    AAT.ToFloat(max_value=255)
                ])
            ]
            
        elif dataset == "Gaussian Noise":
            train_transform = [
                AAT.GaussNoise(var_limit=.01)
            ]
            
        elif dataset == "Speckle Noise":
            train_transform = [
                AAT.MultiplicativeNoise(multiplier=(0.8,1.2), elementwise=True)
            ]
            
        elif dataset == "Speckle Filter - eLee":
            train_transform = [
                
            ]
            
        elif dataset == "Speckle Filter - Frost":
            train_transform = [
                
            ]
            
        elif dataset == "Speckle Filter - GMAP":
            train_transform = [
                
            ]
            
        elif dataset == "Light Pixel":
            train_transform = [
                albu.MotionBlur(blur_limit=11),
                AAT.Sharpen(alpha=(0.1,0.4), lightness=(.9,1.0)),
                AAT.GaussNoise(var_limit=.01)
            ]
            
        elif dataset == "Light Geometry":
            train_transform = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10, p=0.5, border_mode=0),
                albu.Affine(shear={'y': (-10, 10)}, p=0.5)
            ]
            
        elif dataset == "Heavy Geometry":
            train_transform = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=20, p=0.5, border_mode=0),
                albu.Affine(shear={'x': (-10, 10)}, p=0.5),
                albu.Affine(shear={'y': (-10, 10)}, p=0.5),
                albu.augmentations.CoarseDropout(max_holes=6, max_height=40,
                    max_width=40, min_holes=2, min_height=30, min_width=30,
                    fill_value=0.0, mask_fill_value=0)
            ]
            
        elif dataset == "Combination":
            train_transform = [
                albu.MotionBlur(blur_limit=11),
                AAT.Sharpen(alpha=(0.1,0.4), lightness=(.9,1.0)),
                AAT.GaussNoise(var_limit=.01),
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10, p=0.5, border_mode=0),
                albu.Affine(shear={'y': (-10, 10)}, p=0.5)
            ]
            
        return albu.Compose(train_transform)

    def get_validation_augmentation():
        test_transform = [

        ]
        return albu.Compose(test_transform)

    ENCODER = 'tu-efficientnet_b4'
    ENCODER_WEIGHTS = None # 'imagenet', 'pre-trained from:..', None
    CLASSES = ['building']
    ACTIVATION = 'sigmoid' 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    ).to(DEVICE)
    
#▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲#

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(dataset), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class CombinedLoss(nn.Module):
        def __init__(self, loss_a, loss_b, weight_a=0.5, weight_b=0.5):
            super(CombinedLoss, self).__init__()
            self.loss_a = loss_a
            self.loss_b = loss_b
            self.weight_a = weight_a
            self.weight_b = weight_b
            self.__name__ = loss_a.__class__.__name__ + '_' + loss_b.__class__.__name__ 

        def forward(self, output, target):
            return self.weight_a * self.loss_a(output, target) + self.weight_b * self.loss_b(output, target)

    class DiceScore(base.Metric):
        __name__ = "dice_score"

        def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
            super().__init__(**kwargs)
            self.eps = eps
            self.threshold = threshold
            self.activation = Activation(activation)
            self.ignore_channels = ignore_channels

        def forward(self, y_pr, y_gt):
            y_pr = self.activation(y_pr)
            return F.f_score(
                y_pr,
                y_gt,
                eps=self.eps,
                beta=1.0,  
                threshold=self.threshold,
                ignore_channels=self.ignore_channels,
            )

    DiceLoss = utils.losses.DiceLoss()
    CE_Loss = torch.nn.CrossEntropyLoss()
    combined_criterion = CombinedLoss(DiceLoss, CE_Loss, weight_a=0.5, weight_b=0.5)

    metrics = [
        DiceScore(),
        IoU(),
    ]

    optimizer = torch.optim.AdamW([
        dict(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    ])

    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00005)
    
    class CustomCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
            self.T_max = T_max
            self.eta_min = eta_min
            super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (0.5 * (1 + math.cos(math.pi * self.last_epoch / self.T_max)))
                    for base_lr in self.base_lrs]

    scheduler = CustomCosineAnnealingLR(optimizer, T_max=epochs)

    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=combined_criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=combined_criterion, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    epochs = epochs
    save_interval = 10  

    max_score = 0

    save_dir = os.path.join(workspace_path, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    config = wandb.config

    train_cumulative_dice_score = 0.0
    train_cumulative_loss = 0.0

    max_dice_score = 0

    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # 에폭마다 학습률 업데이트
        scheduler.step()

        # Weights & Biases 로깅
        wandb.log({
            "Train Dice Score": train_logs['dice_score'],
            "Train IoU Score": train_logs['iou_score'],
            "Train Loss": train_logs['DiceLoss_CrossEntropyLoss'],
            "Valid Dice Score": valid_logs['dice_score'],
            "Valid IoU Score": valid_logs['iou_score'],
            "Valid Loss": valid_logs['DiceLoss_CrossEntropyLoss'],
            "Learning Rate": optimizer.param_groups[0]['lr']
        })

        if valid_logs['dice_score'] > max_dice_score:
            max_dice_score = valid_logs['dice_score']
            torch.save(model, os.path.join(save_dir, f'best_model_epoch_{epoch}.pth'))

    wandb.finish()