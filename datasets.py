from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets
import numpy as np
import pydicom
import json
import os
from dataclasses import dataclass
from transforms import toTensor, GetImagesOnly

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

def prepocess_xray(x):
    n_bins = 2 ** 12
    x = x / n_bins - 0.5
    return x



def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
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

def get_XRAY(augment, dataroot, download):

    image_shape = (128, 128, 1)
    num_classes = 1

    # if augment:
    #     transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    # else:
    #     transformations = []

    create_transform=transforms.Compose([GetImagesOnly, toTensor, prepocess_xray, add_labels] )

    base_dir = dataroot # r"/share/colon/cviviers/artifact/data/Dicom/"
    json_path = os.path.join(base_dir, "data_description_clock.json")

    xray_dataset = ComposedXrayImageDataset(base_dir, json_path, None, series_per_mode=1, overlap=0.1, patch_size=128, 
                                            images_per_series=200, transform=create_transform, modes_to_exclude=list(np.arange(0, 17)))


    # split dataset into train, val and test 80/10/10
    train_size = int(0.8 * len(xray_dataset))
    val_size = int(0.1 * len(xray_dataset))
    test_size = len(xray_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(xray_dataset, [train_size, val_size, test_size])

    # train_dataset = datasets.ImageFolder(
    return image_shape, num_classes, train_dataset, val_dataset
    
def add_labels(x):
    return (x, 1)

# data class holding each image, the fluoro flavor, exposure/fluoroscopy, source-image distance
@dataclass
class FilteredImageData:
    low_img: np.ndarray
    high_img: np.ndarray
    fluoro_flavor: np.int8
    exposure: np.bool8
    sid: np.int16

    def __init__(self, low_img, high_img, fluoro_flavor, exposure, sid):
        self.low_img = low_img
        self.high_img = high_img
        self.fluoro_flavor = fluoro_flavor
        self.exposure = exposure
        self.sid = sid

@dataclass
class ImageData:
    img: np.ndarray
    fluoro_flavor: np.int8
    exposure: np.bool8
    sid: np.int16

    def __init__(self, img, fluoro_flavor, exposure, sid):

        self.img = img
        self.fluoro_flavor = fluoro_flavor
        self.exposure = exposure
        self.sid = sid

def get_code(flavor, exposure, sid):
    
    # •	Flavor: first, SID 90
    # •	Flavor: second, SID 90
    # •	Flavor: third, SID 90
    # •	Flavor: first, SID 100
    # •	Flavor: second, SID 100
    # •	Flavor: third, SID 100
    # •	Flavor: first, SID 110
    # •	Flavor: second, SID 110
    # •	Flavor: third, SID 110

    # •	Exposure: first, SID 90
    # •	Exposure: second, SID 90
    # •	Exposure: third, SID 90
    # •	Exposure: first, SID 100
    # •	Exposure: second, SID 100
    # •	Exposure: third, SID 100
    # •	Exposure: first, SID 110
    # •	Exposure: second, SID 110
    # •	Exposure: third, SID 110
    
    sid = int(sid/10)

    if flavor == 0 and exposure == 0 and sid == 90:
        return 0
    elif flavor == 1 and exposure == 0 and sid == 90:
        return 1
    elif flavor == 2 and exposure == 0 and sid == 90:
        return 2
    elif flavor == 0 and exposure == 0 and sid == 100:
        return 3
    elif flavor == 1 and exposure == 0 and sid == 100:
        return 4
    elif flavor == 2 and exposure == 0 and sid == 100:
        return 5
    elif flavor == 0 and exposure == 0 and sid == 110:
        return 6
    elif flavor == 1 and exposure == 0 and sid == 110:
        return 7
    elif flavor == 2 and exposure == 0 and sid == 110:
        return 8
    elif flavor == 0 and exposure == 1 and sid == 90:
        return 9
    elif flavor == 1 and exposure == 1 and sid == 90:
        return 10
    elif flavor == 2 and exposure == 1 and sid == 90:
        return 11
    elif flavor == 0 and exposure == 1 and sid == 100:
        return 12
    elif flavor == 1 and exposure == 1 and sid == 100:
        return 13
    elif flavor == 2 and exposure == 1 and sid == 100:
        return 14
    elif flavor == 0 and exposure == 1 and sid == 110:
        return 15
    elif flavor == 1 and exposure == 1 and sid == 110:
        return 16
    elif flavor == 2 and exposure == 1 and sid == 110:
        return 17
    else:
        return -1





class ComposedXrayImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, json_description, image_filter=None, transform = None, patch_size: int = 128, overlap: float = 0.25, series_per_mode=6, images_per_series = None, preload=True, modes_to_exclude = []):

        self.base_dir = base_dir
        self.transform = transform
        self.overlap = overlap
        self.patch_size = patch_size
        self.data = list()
        self.filter = image_filter
        self.series_per_mode = series_per_mode

        
        data_description = open(json_description)
        data_description = json.load(data_description)

        total_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]

        print(base_dir)

        if preload:

            for item in data_description.keys():
                series_data = data_description[item]

                flavor, exposure, sid = series_data['flavor'], series_data['fluo/exposure'], series_data['sid']
                s = get_code(flavor, exposure, sid)
                if s in modes_to_exclude:
                    continue
                if total_count[s] < series_per_mode:
                    total_count[s] += 1
                    print(f'Adding series {s}')
                    path = base_dir + series_data['path']
                    series = pydicom.dcmread(path)
                    series_images = series.pixel_array
                    _, img_h, img_w  = series_images.shape

                    X_points = self.start_points(img_w, self.patch_size, self.overlap)
                    Y_points = self.start_points(img_h, self.patch_size, self.overlap)
                    
                    for idx, image in enumerate(series_images):
                        if images_per_series is not None and idx >= images_per_series:
                            break
                        # image = image/np.power(2, 12) # Normalize images between 0 and 1
                        if self.filter is not None:
                            low, high = self.filter(image)

                            low_patches, count = self.split_image_with_points(low, X_points, Y_points)
                            high_patches, count = self.split_image_with_points(high, X_points, Y_points)

                            for patch_idx in range(count):
                                self.data.append(FilteredImageData(low_img = low_patches[patch_idx], high_img = high_patches[patch_idx], fluoro_flavor = flavor , exposure = exposure, sid = sid))

                        else:
                            image_patches, count = self.split_image_with_points(image, X_points, Y_points)

                            for patch_idx in range(count):
                                self.data.append(ImageData(img = image_patches[patch_idx], fluoro_flavor = flavor , exposure = exposure, sid = sid))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx]


    def split_image(self, image):
        img_h, img_w, img_c,  = image.shape
        X_points = self.start_points(img_w, self.patch_size, self.overlap)
        Y_points = self.start_points(img_h, self.patch_size, self.overlap)
        count = 0
        frmt = "png"
        patches = np.empty((len(Y_points)*len(X_points), self.patch_size, self.patch_size, img_c))
        for i in Y_points:
            for j in X_points:
                split = image[i:i+self.patch_size, j:j+self.patch_size, :]
                patches[count,:, :, :] = split
                #cv2.imwrite('patch_{}.{}'.format( count, frmt), split)
                count += 1
        return patches

    def split_image_with_points(self, image, X_points, Y_points):

        count = 0

        patches = np.empty((len(Y_points)*len(X_points), self.patch_size, self.patch_size))
        for i in Y_points:
            for j in X_points:
                split = image[i:i+self.patch_size, j:j+self.patch_size]
                patches[count,:, :] = split
                #cv2.imwrite('patch_{}.{}'.format( count, frmt), split)
                count += 1
        return patches, count

    
    def start_points(self, size, patch_size, overlap=0):
        points = [0]
        stride = int(patch_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + patch_size >= size:
                points.append(size - patch_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points
    


