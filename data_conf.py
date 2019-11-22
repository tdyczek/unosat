from pathlib import Path
from typing import List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features


class Image:
    def __init__(self, img_path: Path):
        self.vv = list(img_path.glob('**/*vv*.tif'))[0]
        self.vh = list(img_path.glob('**/*vh*.tif'))[0]

    def __repr__(self):
        return f"({self.vv.name}, " \
               f"{self.vh.name})\n"


class CityData:
    def __init__(self, city_dir: Path, with_mask=True):
        self.name = city_dir.name
        self.images = [Image(image_dir) for image_dir in
                       city_dir.glob("*") if image_dir.is_dir()]
        if with_mask:
            self.mask = list(city_dir.glob('*.shp'))[0]

    def __repr__(self):
        return f"{self.name}\n" \
               f"    {self.mask}\n" \
               f"    {[image for image in self.images]}"


def extract_ims(cities_path: Path, with_mask=True):
    cities_dirs = cities_path.glob("*_2015")
    cities = np.array([CityData(city_dir, with_mask)
                       for city_dir in cities_dirs])
    return cities


def save_city_mask(city: CityData, out_dir: Path):
    shp_mask = gpd.read_file(city.mask).to_crs({'init': 'epsg:32638'})
    shp_mask['positive'] = 1
    meta = rasterio.open(city.images[0].vv).meta.copy()
    meta.update(dtype="int16")
    with rasterio.open(out_dir / f'{city.name}.tif', 'w+', **meta) as out:
        out_arr = out.read(1)

        shapes = ((geom, value) for geom, value in zip(shp_mask.geometry, shp_mask.positive))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)


if __name__ == '__main__':
    cities = extract_ims(Path("data/train/"))
    for city in cities:
        save_city_mask(city, Path('data/train/masks'))
