import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ImageDataset import ImageDataset

# DO NOT CHANGE:
crop_size = 224 # Required by resnet152

# See https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
def collate_fn (data):
    # Sort datalist by caption length; descending order
    data.sort(key = lambda data_pair: len(data_pair[1]), reverse=True)
    images, captions = zip(*data)
    
    # Merge images (from tuple of 3D Tensor to 4D Tensor)
    images = torch.stack(images, 0)
    
    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(caption) for caption in captions] # List of caption lengths
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, caption in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = caption[:end]
        
    return images, targets, lengths

# Transform to modify images for pre-trained ResNet base
transform = transforms.Compose([
    transforms.Resize((crop_size, crop_size)), # Match resnet size
    transforms.ToTensor(),
    # See for magic #'s: http://pytorch.org/docs/master/torchvision/models.html
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def getDataLoader(data_dir, batch_size, shuffle=True, num_workers=2):
    dataset = ImageDataset(data_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return data_loader, dataset.raw_image_names