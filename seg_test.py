import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import albumentations as albu
from tqdm import tqdm

def load_model(model_path, device):
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    return model

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=lambda x: x.transpose(2, 0, 1).astype('float32')),
    ]
    return albu.Compose(_transform)

def predict(model, image, preprocessing):
    with torch.no_grad():
        sample = preprocessing(image=image)['image']
        sample = torch.from_numpy(sample).to(device).unsqueeze(0)
        prediction = model(sample)
        prediction = prediction.squeeze().cpu().numpy()
    return prediction

def main(model_path, test_images_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    encoder = 'tu-efficientnet_b0'
    encoder_weights = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    model = load_model(model_path, device)
    preprocessing = get_preprocessing(preprocessing_fn)

    test_images = os.listdir(test_images_dir)
    for image_name in tqdm(test_images):
        image_path = os.path.join(test_images_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = predict(model, image, preprocessing)
        
        # Optionally, post-process mask and save
        mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding
        cv2.imwrite(f'predicted_masks/{image_name}', mask)

if __name__ == '__main__':
    model_path = 'path/to/saved_model.pth'
    test_images_dir = 'path/to/test_images'
    main(model_path, test_images_dir)