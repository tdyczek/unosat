from pathlib import Path
from typing import List
import numpy as np
from skimage.io import imread
import random

from data_conf import CityData
from albumentations import Compose, RandomRotate90, VerticalFlip, \
    HorizontalFlip, Transpose, ShiftScaleRotate, RandomSizedCrop

transform = Compose([
    RandomRotate90(),
    VerticalFlip(),
    HorizontalFlip(),
    Transpose()
])


def clip_perc(im, l=5, u=95):
    l_perc = np.percentile(im, l)
    u_perc = np.percentile(im, u)
    return np.clip(im, l_perc, u_perc)


def load_city_imagery(city: CityData, v_coef=(0.124, 0.129), h_coef=(0.0212, 0.0216)):
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


class TrainDataset:
    def __init__(self, cities: List[CityData], mask_dir: Path, set_len=3500, im_size=448, augment=False):
        self.imagery = [load_city_imagery(city) for city in cities]
        self.masks = [load_mask(city, mask_dir) for city in cities]
        self.im_size = im_size
        self.set_len = set_len
        self.augment = augment

    def __len__(self):
        return self.set_len

    def __getitem__(self, ix):
        city_ix = ix % len(self.imagery)
        image_ix = ix % 4

        vv, vh = self.imagery[city_ix][image_ix]
        mask = self.masks[city_ix]

        vv, vh, y = cut_data(vv, vh, mask, self.im_size)
        x = np.stack([vv, vh, vv - vh], -1)

        if self.augment:
            data = transform(image=x, mask=y)
            x, y = data['image'], data['mask']

        return x.transpose([2, 0, 1])[:2, ...], y
