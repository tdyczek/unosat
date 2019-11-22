from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes as r_shapes
from tqdm import tqdm

from data_conf import extract_ims
from data_provider import load_city_imagery
from models import LinkNet

MODELS_PATHS = ['data/models/2/Mosul_2015/linknet',
                'data/models/2/Najaf_2015/linknet',
                'data/models/2/Nasiryah_2015/linknet',
                'data/models/2/Souleimaniye_2015/linknet']
TEST_DATA = Path("data/test")
OUT_PATH = Path("data/out/2")
WINDOW_SIZE = 2240


def get_model(path):
    model = LinkNet().cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def infer_one(image, model, out_image):
    for x0 in range(0, image[0].shape[0], WINDOW_SIZE):
        for x1 in range(0, image[0].shape[1], WINDOW_SIZE):
            if x0 + WINDOW_SIZE >= image[0].shape[0]:
                x0 = image[0].shape[0] - WINDOW_SIZE

            if x1 + WINDOW_SIZE >= image[0].shape[1]:
                x1 = image[0].shape[1] - WINDOW_SIZE

            chunk = np.stack([image[0][x0: x0 + WINDOW_SIZE, x1: x1 + WINDOW_SIZE],
                              image[1][x0: x0 + WINDOW_SIZE, x1: x1 + WINDOW_SIZE]], axis=0)
            chunk = chunk[np.newaxis, ...]
            with torch.no_grad():
                chunk = torch.tensor(chunk).cuda()
                pred = model(chunk)[0, 0].detach().cpu().numpy()
            out_image[x0: x0 + WINDOW_SIZE, x1: x1 + WINDOW_SIZE] += pred
    return out_image


def infer_city(city_ims, models):
    out_shape = city_ims[0][0].shape
    out_im = np.zeros(out_shape, dtype=np.float32)
    for model in tqdm(models):
        for im in tqdm(city_ims):
            out_im = infer_one(im, model, out_im)
    return out_im


def main():
    models = list(map(get_model, MODELS_PATHS))
    cities_paths = extract_ims(TEST_DATA, with_mask=False)
    cities_ims = [load_city_imagery(city) for city in cities_paths]
    for city, ims in zip(cities_paths, cities_ims):
        print(f'Infering {city.name}')
        image = infer_city(ims, models)
        with rasterio.open(city.images[0].vv) as src:
            results = (
                {'properties': {'id': 1}, 'geometry': s}
                for i, (s, v) in enumerate(r_shapes((image > 0).astype(np.uint8), mask=(image > 0), transform=src.transform)))
        geoms = list(results)
        g_df = gpd.GeoDataFrame.from_features(geoms)
        g_df.crs = {'init': 'epsg:32638'}
        g_df.to_file(OUT_PATH / f'{city.name}.shp')


if __name__ == '__main__':
    with torch.no_grad():
        main()
