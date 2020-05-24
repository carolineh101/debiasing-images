from math import ceil
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torch
import numpy as np
import pdb
import random


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


def load_celeba(dir_name='data/CelebA/', splits=['train', 'valid', 'test'], subset_percentage=1, protected_percentage=1, balance_protected=True,
                batch_size=32, num_workers=1):
    """
    Return DataLoaders for CelebA, downloading dataset if necessary.

    Params:
    - subset_percentage: Percentage of dataset to include in dataloaders.
    """
    data_loaders = {} # right now, it seems our three splits will return three dataloaders of the same size? 
    for split in splits:
        dataset = CelebADataset(split=split, subset_percentage=subset_percentage, protected_percentage=protected_percentage, \
            balance_protected=balance_protected)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        data_loaders[split] = data_loader
    return data_loaders


class CelebADataset(Dataset):

    def __init__(self, split, dir_name='data/', subset_percentage = 1, protected_percentage = 1, balance_protected=True):
        self.transform_image = transform_image
        self.gender_index = 20
        self.dataset = datasets.CelebA(
            dir_name,
            split=split,
            transform=transform_image,
            target_transform=None,
            download=True
        )

        if subset_percentage < 1:
            self.dataset = Subset(self.dataset, range(ceil(subset_percentage * len(self.dataset))))
        
        # Handle protected split (only relevant for train).
        self.protected = np.zeros(len(self.dataset))
        if split == 'train':
            if balance_protected:
                # Get gender information.
                genders = np.array([self.dataset[i][1][20] for i in range(len(self.dataset))])
                male_idxs = np.where(genders == 1)[0].tolist()
                female_idxs = np.where(genders == 0)[0].tolist()
                # Set number of protected data points.
                max_percentage = min(len(male_idxs), len(female_idxs)) * 2.0 / len(self.dataset)
                if protected_percentage > max_percentage: protected_percentage = max_percentage
                num_protected = ceil(protected_percentage * len(self.dataset))
                if num_protected % 2 == 1: num_protected -= 1
                # Create protected split.
                self.protected_split = random.sample(male_idxs, int(num_protected / 2))
                self.protected_split.extend(random.sample(female_idxs, int(num_protected / 2)))
                self.protected[self.protected_split] = 1
            else:
                num_protected = ceil(protected_percentage * len(self.dataset))
                self.protected_split = random.sample(range(len(self.dataset)), num_protected)
                self.protected[self.protected_split] = 1    


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, targets = self.dataset[idx]

        gender = targets[self.gender_index]
        targets = torch.cat((targets[:self.gender_index], targets[self.gender_index+1:]))

        # [batch_size] -> [batch_size, 1]
        gender = gender.unsqueeze(-1)
        protected = self.protected[idx]

        return image, targets, gender, protected
