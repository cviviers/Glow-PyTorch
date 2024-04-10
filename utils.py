import math
import torch
import torchvision
import os
import random
from PIL import Image
import numpy as np

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    dequantized_x = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return dequantized_x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

class CIFAR10C(torchvision.datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None, corruptions_file=None, severity=1):
        corruptions = load_txt(corruptions_file)

        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)[(severity-1)*10000:severity*10000]
        self.targets = np.load(target_path)[(severity-1)*10000:severity*10000]

        
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
        
    return torch.utils.data.Subset(dataset, indices)


class TinyImageNet(torchvision.datasets.VisionDataset):
    # load all images in train/val/test
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.data = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.load_data()

    def load_data(self):

        if self.split == 'train':
            data_path = os.path.join(self.root, 'train')

            # find all the images in the train directory
            for class_name in os.listdir(data_path):
                class_path = os.path.join(data_path, class_name, 'images')
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    # read image as rgb
                    img = np.array(Image.open(img_path))
                    # if not rgb, convert to rgb
                    if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                    self.data.append(img)
                    self.targets.append(class_name)

        elif self.split == 'val':
            data_path = os.path.join(self.root, 'val')
            with open(os.path.join(data_path, 'val_annotations.txt')) as f:
                for line in f:
                    img_name, class_name = line.split('\t')[:2]
                    img_path = os.path.join(data_path, 'images', img_name)
                    # temp open image

                    img = np.array(Image.open(img_path))
                    if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                    self.data.append(img)
                    self.targets.append(class_name)
        elif self.split == 'test':
            data_path = os.path.join(self.root, 'test', 'images')
            for img_name in os.listdir(data_path):
                img_path = os.path.join(data_path, img_name)
                img = np.array(Image.open(img_path))
                if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                self.data.append(img)
                self.targets.append('')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets