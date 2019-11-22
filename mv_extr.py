from skimage.io import imread
import numpy as np
from pathlib import Path


def clip_perc(im, l=5, u=95):
    l_perc = np.percentile(im, l)
    u_perc = np.percentile(im, u)
    return np.clip(im, l_perc, u_perc)


if __name__ == '__main__':
    print("vv")
    ims = [clip_perc(imread(pth)).ravel() for pth in Path('.').glob('**/*vv*.tif')]
    ims = np.concatenate(ims, axis=None)
    print(np.mean(ims), np.std(ims))

    print("vh")
    ims = [clip_perc(imread(pth)).ravel() for pth in Path('.').glob('**/*vh*.tif')]
    ims = np.concatenate(ims, axis=None)
    print(np.mean(ims), np.std(ims))
