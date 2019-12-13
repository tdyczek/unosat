from pathlib import Path

MODELS_PATHS = [
    "data/models/4/Mosul_2015/unet",
    "data/models/4/Najaf_2015/unet",
    "data/models/4/Nasiryah_2015/unet",
    "data/models/4/Souleimaniye_2015/unet",
]

TEST_DATA = Path("data/test")
OUT_PATH = Path("data/out/6")

TRAIN_PATH = Path("data/train")
MASK_DIR = Path("data/train/masks")
MODELS_DIR = Path("data/models/4/")
