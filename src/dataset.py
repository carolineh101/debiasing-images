from math import ceil
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Transform to modify images for pre-trained ResNet base
# See for magic #'s: http://pytorch.org/docs/master/torchvision/models.html
def transform_image(image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Transform image.
    """
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    return transform


def target_transform(target, attr_idx=20):
    """
    Transform target labels to remove attribute at given index.
    Note: 20 is index for gender.
    """
    for i in range(len(target)):
        target[i] = torch.cat(target[i][:attr_idx], target[i][attr_idx+1:])
    return target


def load_celeba(dir_name='data/CelebA/', splits=['train', 'valid', 'test'], subset_percentage=1,
                batch_size=32, num_workers=2):
    """
    Return DataLoaders for CelebA, downloading dataset if necessary.

    Params:
    - subset_percentage: Percentage of dataset to include in dataloaders.
    """
    data_loaders = {}
    for split in splits:
        dataset = datasets.CelebA(dir_name + split + '/', split=split, transform=transform_image,
                                  target_transform=target_transform, download=True)
        if subset_percentage < 1:
            dataset = Subset(dataset, range(ceil(subset_percentage * len(dataset))))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        data_loaders[split] = data_loader
    return data_loaders
