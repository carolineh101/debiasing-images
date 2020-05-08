from math import ceil
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torch
import numpy as np
import pdb


# Transform to modify images for pre-trained ResNet base
# See for magic #'s: http://pytorch.org/docs/master/torchvision/models.html
def transform_image(image, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Transform image.
    """
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    return transform(image)


def target_transform(target, attr_idx=20):
    """
    Transform target labels to remove attribute at given index.
    Note: 20 is index for gender.
    """
    return torch.cat((target[:attr_idx], target[attr_idx+1:]))


def load_celeba(dir_name='data/CelebA/', splits=['train', 'valid', 'test'], subset_size = -1,
                batch_size=32, num_workers=1):
    """
    Return DataLoaders for CelebA, downloading dataset if necessary.

    Params:
    - subset_percentage: Percentage of dataset to include in dataloaders.
    """
    data_loaders = {} # right now, it seems our three splits will return three dataloaders of the same size? 
    for split in splits:
        dataset = CelebADataset(split=split, subset_size=subset_size)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_loaders[split] = data_loader
    return data_loaders


class CelebADataset(Dataset):

    def __init__(self, split, dir_name='data/CelebA/', subset_size = 1):
        self.transform_image = transform_image
        self.target_transform = target_transform
        self.dataset = datasets.CelebA(dir_name + split + '/', split=split, transform=transform_image,
                                target_transform = target_transform, download=True)

        if subset_size > 0:
            train_indices = torch.from_numpy(np.random.choice(len(self.dataset), size=subset_size, replace=False)) 
            self.dataset = Subset(self.dataset, train_indices)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, targets = self.dataset[idx]

        gender_index = 20
        targets = torch.cat((targets[:gender_index], targets[gender_index+1:]))
        gender = targets[gender_index]

        return image, targets, gender



