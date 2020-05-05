import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset (Dataset):
    def __init__ (self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.raw_image_names = []
        
    def __len__ (self):
        return len(self.raw_image_names)
    
    def __getitem__ (self, idx):
        img_path = self.raw_image_names[idx]
        
        # Get image from filesystem
        image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        image = self.transform(image)
        
        return image