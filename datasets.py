from pathlib import Path
import os
import torch
import torch.nn.functional as F

from torchvision import transforms, datasets
from utils import TinyImageNet, SSBHard, NINCO, iNaturalist, Texture, OpenImageOOD, MNISTC

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


def one_hot_encode(target, num_classes=10):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)

    return one_hot_encoding

def one_hot_encode_100(target, num_classes=100):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)

    return one_hot_encoding


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

def get_CIFAR100(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 100

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "CIFAR100"
    train_dataset = datasets.CIFAR100(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode_100,
        download=download,
    )

    test_dataset = datasets.CIFAR100(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode_100,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

def get_MNIST(augment, dataroot, download):
    image_shape = (32, 32, 1)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            
        ]
    else:
        transformations = []

    transformations.extend([transforms.Resize(image_shape[0]), transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.Resize(image_shape[0]), transforms.ToTensor(), preprocess])

    path = Path(dataroot) / "data" / "MNIST"
    train_dataset = datasets.MNIST(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )
    # split into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    test_dataset = datasets.MNIST(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, val_dataset, test_dataset


def get_TinyImageNet(augment, dataroot, download):
    image_shape = (64, 64, 3)
    num_classes = 200

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    # pass one_hot_encode as target_transform

    path = Path(dataroot) 
    train_dataset = TinyImageNet(
        path,
        split= "train",
        transform=train_transform,
        target_transform=None,
    )

    val_dataset = TinyImageNet(
        path,
        split = "val",
        transform=test_transform,
        target_transform=None,
    )

    if os.path.exists(path / dataroot / "test"):
        test_dataset = TinyImageNet(
            path,
            split="test",
            transform=test_transform,
            target_transform=None,
        )
        return image_shape, num_classes, train_dataset, val_dataset, test_dataset
    else:

        return image_shape, num_classes, train_dataset, val_dataset
    

def get_TinyImageNet32(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 200

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.Resize(size=(32,32)), transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.Resize(size=(32,32)), transforms.ToTensor(), preprocess])

    # pass one_hot_encode as target_transform

    path = Path(dataroot) 
    train_dataset = TinyImageNet(
        path,
        split= "train",
        transform=train_transform,
        target_transform=None,
    )

    val_dataset = TinyImageNet(
        path,
        split = "val",
        transform=test_transform,
        target_transform=None,
    )

    if os.path.exists(path / dataroot / "test"):
        test_dataset = TinyImageNet(
            path,
            split="test",
            transform=test_transform,
            target_transform=None,
        )
        return image_shape, num_classes, train_dataset, val_dataset, test_dataset
    else:

        return image_shape, num_classes, train_dataset, val_dataset
    
def get_SSBHard(augment, dataroot, download):

    image_shape = (64, 64)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_ssb_hard.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = SSBHard(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_NINCO(augment, dataroot, download):

    image_shape = (64, 64)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_ninco.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = NINCO(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_iNaturalist(augment, dataroot, download):
    
    image_shape = (64, 64)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_inaturalist.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = iNaturalist(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_Texture(augment, dataroot, download):
        
    image_shape = (64, 64)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_dtd_(OTHER).txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = Texture(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_Texture32(augment, dataroot, download):
        
    image_shape = (32, 32)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_dtd_(OTHER).txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = Texture(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_Texture32_Grey(augment, dataroot, download):
        
    image_shape = (32, 32)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_dtd_(OTHER).txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), preprocess])

    test_dataset = Texture(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_PLACES365(augment, dataroot, download):

    image_shape = (32, 32)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_places365.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = iNaturalist(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_PLACES365_grey(augment, dataroot, download):

    image_shape = (32, 32)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_places365.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), transforms.Grayscale(num_output_channels=1), preprocess])

    test_dataset = iNaturalist(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_OpenImageOOD(augment, dataroot, download):

    image_shape = (64, 64)
    num_classes = 200
    splits_file = r'/media/chris/My Passport/Philips/Anomaly/openOOD/test_openimage_o.txt'

    
    test_transform = transforms.Compose([transforms.Resize(image_shape), transforms.ToTensor(), preprocess])

    test_dataset = OpenImageOOD(dataroot, transform=test_transform, target_transform=None, data_to_load=splits_file)

    # only load the data from the file

    
    return image_shape, num_classes, test_dataset

def get_MNIST_RGB(augment, dataroot, download):
    image_shape = (32, 32)
    num_classes = 10
    test_transform = transforms.Compose([transforms.Resize(image_shape),  transforms.ToTensor(), GreyToRGB(), preprocess])
    mnist_test = datasets.MNIST(root=dataroot, train=False, download=True, transform=test_transform, target_transform=one_hot_encode)
    return image_shape, num_classes, mnist_test



class GreyToRGB():
    def __call__(self, x):
        return x.repeat(3, 1, 1)
    
class SwapChannels():
    # channels first to channels last
    def __call__(self, x):
        return x.permute(1, 2, 0)
    
class CustomPermtation():
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, x):
        
        return x.permute(2, 0, 1)

def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset
