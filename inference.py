from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes as r_shapes
from skimage.transform import resize
from tqdm import tqdm

from conf import OUT_PATH, MODELS_PATHS, TEST_DATA
from constants import INFERENCE_PRED_WINDOW, BORDER, INFERENCE_CONTEXT_WINDOW
from data_conf import extract_ims
from data_provider import load_city_imagery
from models import UNet11


def get_model(path):
    model = UNet11().cuda()
    model.load_state_dict(torch.load(path))
    model = model.eval()
    return model


def infer_one(image, model):
    out_image = np.zeros(image[0].shape, dtype=np.float32)
    vv, vh = image[0], image[1]
    for x0 in range(0, vv.shape[0], INFERENCE_PRED_WINDOW):
        for x1 in range(0, vv.shape[1], INFERENCE_PRED_WINDOW):
            if x0 + INFERENCE_PRED_WINDOW >= vv.shape[0]:
                x0 = vv.shape[0] - INFERENCE_PRED_WINDOW
            if x1 + INFERENCE_PRED_WINDOW >= vv.shape[1]:
                x1 = vv.shape[1] - INFERENCE_PRED_WINDOW

            if x0 - BORDER < 0:
                b0 = 0
            elif x0 + INFERENCE_PRED_WINDOW + BORDER >= vv.shape[0]:
                b0 = vv.shape[0] - INFERENCE_CONTEXT_WINDOW
            else:
                b0 = x0 - BORDER

            if x1 - BORDER < 0:
                b1 = 0
            elif x1 + INFERENCE_PRED_WINDOW + BORDER >= vv.shape[1]:
                b1 = vv.shape[1] - INFERENCE_CONTEXT_WINDOW
            else:
                b1 = x1 - BORDER
            chunk = np.stack(
                [
                    vv[
                        b0 : b0 + INFERENCE_CONTEXT_WINDOW,
                        b1 : b1 + INFERENCE_CONTEXT_WINDOW,
                    ],
                    vh[
                        b0 : b0 + INFERENCE_CONTEXT_WINDOW,
                        b1 : b1 + INFERENCE_CONTEXT_WINDOW,
                    ],
                ],
                axis=0,
            )

            chunk = chunk[np.newaxis, ...]
            chunk = torch.tensor(chunk).cuda().float()
            pred = model(chunk)[0, 0].detach().cpu().numpy()
            out_image[
                x0 : x0 + INFERENCE_PRED_WINDOW, x1 : x1 + INFERENCE_PRED_WINDOW
            ] += pred[
                x0 - b0 : x0 - b0 + INFERENCE_PRED_WINDOW,
                x1 - b1 : x1 - b1 + INFERENCE_PRED_WINDOW,
            ]
    return out_image


def infer_city(city_ims, models):
    out_shape = city_ims[0][0].shape
    out_image = np.zeros(out_shape, dtype=np.float32)
    for model in tqdm(models):
        for im in tqdm(city_ims):
            single_out = infer_one(im, model)
            if (single_out.shape[0] != out_shape[0]) or (
                single_out.shape[1] != out_shape[1]
            ):
                single_out = resize(single_out, out_shape, preserve_range=True)
            out_image += single_out
    return out_image


def main():
    models = list(map(get_model, MODELS_PATHS))
    cities_paths = extract_ims(TEST_DATA, with_mask=False)
    cities_ims = [load_city_imagery(city) for city in cities_paths]
    for city, ims in zip(cities_paths, cities_ims):
        print(f"Infering {city.name}")
        image = infer_city(ims, models)
        with rasterio.open(city.images[0].vv) as src:
            results = (
                {"properties": {"id": 1}, "geometry": s}
                for i, (s, v) in enumerate(
                    r_shapes(
                        (image > 0).astype(np.uint8),
                        mask=(image > 0),
                        transform=src.transform,
                    )
                )
            )
        geoms = list(results)
        g_df = gpd.GeoDataFrame.from_features(geoms)
        g_df.crs = {"init": "epsg:32638"}
        g_df.to_file(OUT_PATH / f"{city.name}.shp")


if __name__ == "__main__":
    with torch.no_grad():
        main()
