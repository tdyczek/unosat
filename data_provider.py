from math import ceil
from pathlib import Path
from typing import List
import numpy as np
from skimage.io import imread
import random

from constants import TRAIN_WINDOW
from data_conf import CityData
from albumentations import Compose, RandomRotate90, VerticalFlip, \
    HorizontalFlip, Transpose, ShiftScaleRotate, RandomSizedCrop, \
    GridDistortion, ShiftScaleRotate, RandomSizedCrop

transform = Compose([
    RandomRotate90(),
    VerticalFlip(),
    HorizontalFlip(),
    Transpose(),
    GridDistortion(p=0.1),
    ShiftScaleRotate(p=0.1),
    RandomSizedCrop((int(0.65*TRAIN_WINDOW), int(0.9*TRAIN_WINDOW)), TRAIN_WINDOW, TRAIN_WINDOW, p=0.1)
])


def clip_perc(im, l=5, u=95):
    l_perc = np.percentile(im, l)
    u_perc = np.percentile(im, u)
    return np.clip(im, 0, 2)


def load_city_imagery(city: CityData, v_coef=(0.142, 0.216), h_coef=(0.0243, 0.0445)):
    raw_ims = []
    for image in city.images:
        vv = clip_perc(imread(image.vv))
        vv = (vv - v_coef[0]) / v_coef[1]

        vh = clip_perc(imread(image.vh))
        vh = (vh - h_coef[0]) / h_coef[1]
        raw_ims.append((vv, vh))
    return raw_ims


def load_mask(city: CityData, mask_dir: Path):
    mask_path = mask_dir / f"{city.name}.tif"
    return np.asarray(imread(mask_path))


def cut_data(vv, vh, mask, im_size):
    i0 = random.randint(0, vv.shape[0] - im_size)
    i1 = random.randint(0, vv.shape[1] - im_size)

    return vv[i0: i0 + im_size, i1: i1 + im_size], \
           vh[i0: i0 + im_size, i1: i1 + im_size], \
           mask[i0: i0 + im_size, i1: i1 + im_size]


def count_dataset_size(cities, im_size):
    data_size = 0
    for city in cities:
        for image in city:
            data_size += (ceil(image[0].shape[0] / im_size) *
                          ceil(image[0].shape[1] / im_size))
    return data_size


class TrainDataset:
    def __init__(self, cities: List[CityData], mask_dir: Path, im_size=TRAIN_WINDOW):
        self.imagery = [load_city_imagery(city) for city in cities]
        self.masks = [load_mask(city, mask_dir) for city in cities]
        self.im_size = im_size
        self.set_len = count_dataset_size(self.imagery, im_size)

    def __len__(self):
        return self.set_len

    def __getitem__(self, ix):
        city_ix = ix % len(self.imagery)
        image_ix = ix % 4

        vv, vh = self.imagery[city_ix][image_ix]
        mask = self.masks[city_ix]

        vv, vh, y = cut_data(vv, vh, mask, self.im_size)
        x = np.stack([vv, vh, vv - vh], -1)

        data = transform(image=x, mask=y)
        x, y = data['image'], data['mask']

        return x.transpose([2, 0, 1])[:2, ...], y


class TestDataset:
    def __init__(self, cities: List[CityData], mask_dir: Path, im_size=TRAIN_WINDOW):
        self.imagery = [load_city_imagery(city) for city in cities]
        self.masks = [load_mask(city, mask_dir) for city in cities]
        self.im_size = im_size
        self.set_len = count_dataset_size(self.imagery, im_size)
        self.image_gen = self.imagery_generator()

    def __len__(self):
        return self.set_len

    def __getitem__(self, ix):
        vv, vh, y = next(self.image_gen)
        x = np.stack([vv, vh, vv - vh], -1)

        return x.transpose([2, 0, 1])[:2, ...], y

    def imagery_generator(self):
        im_size = self.im_size
        city = self.imagery[0]
        mask = self.masks[0]
        while True:
            for image in city:
                vv, vh = image
                for x0 in range(0, image[0].shape[0], self.im_size):
                    for x1 in range(0, image[0].shape[1], self.im_size):
                        if x0 + self.im_size >= image[0].shape[0]:
                            x0 = image[0].shape[0] - self.im_size

                        if x1 + self.im_size >= image[0].shape[1]:
                            x1 = image[0].shape[1] - self.im_size

                        yield vv[x0: x0 + im_size, x1: x1 + im_size], \
                              vh[x0: x0 + im_size, x1: x1 + im_size], \
                              mask[x0: x0 + im_size, x1: x1 + im_size]
