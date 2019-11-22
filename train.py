from typing import List

from sklearn.model_selection import KFold
from pathlib import Path

from data_conf import extract_ims, CityData
from data_provider import TrainDataset
from metrics import dice_loss, iou, jaccard
from models import UNet11
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from tqdm import tqdm
import torch
import numpy as np

TRAIN_PATH = Path('data/train')
MASK_DIR = Path('data/train/masks')
MODELS_DIR = Path('data/models')


def save_model(model, epoch, model_path):
    model_name = f"UNet16_{epoch}"
    path = model_path / model_name
    torch.save(model.state_dict(), path)


def train(epochs: int, models_dir: Path, x_cities: List[CityData], y_city: List[CityData], mask_dir: Path):
    model = UNet11()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    min_loss = sys.maxsize
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch}, lr {optimizer.param_groups[0]["lr"]}')
        print(f'    Training')

        losses = []
        ious = []
        jaccs = []

        train_data = DataLoader(TrainDataset(x_cities, mask_dir, augment=True),
                                batch_size=8,
                                num_workers=2, shuffle=True)

        batch_iterator = enumerate(train_data)

        model = model.train().cuda()
        for i, (x, y) in tqdm(batch_iterator):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()

            y_real = y.view(-1).float()
            y_pred = model(x)
            y_pred_probs = torch.sigmoid(y_pred).view(-1)
            loss = criterion(y_pred.view(-1), y_real) + \
                   .25 * dice_loss(y_pred_probs, y_real)

            iou_ = iou(y_pred_probs.float(), y_real.byte())
            jacc_ = jaccard(y_pred_probs.float(), y_real)
            ious.append(iou_.item())
            losses.append(loss.item())
            jaccs.append(jacc_.item())

            loss.backward()
            optimizer.step()

        print(f'Loss: {np.mean(losses)}, IOU: {np.mean(ious)}, jacc: {np.mean(jaccs)}')

        model = model.eval().cuda()
        losses = []
        ious = []
        jaccs = []
        test_data = DataLoader(TrainDataset(y_city, mask_dir, 1600),
                               batch_size=8,
                               num_workers=2, shuffle=False)
        batch_iterator = enumerate(test_data)
        for i, (x, y) in tqdm(batch_iterator):
            x = x.cuda()
            y = y.cuda()
            y_real = y.view(-1).float()
            y_pred = model(x)
            y_pred_probs = torch.sigmoid(y_pred).view(-1)
            loss = criterion(y_pred.view(-1), y_real) + \
                   .25 * dice_loss(y_pred_probs, y_real)

            iou_ = iou(y_pred_probs.float(), y_real.byte())
            jacc_ = jaccard(y_pred_probs.float(), y_real)
            ious.append(iou_.item())
            losses.append(loss.item())
            jaccs.append(jacc_.item())
        test_loss = np.mean(losses)
        print(f'Loss: {np.mean(losses)}, IOU: {np.mean(ious)}, jacc: {np.mean(jaccs)}')

        scheduler.step(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            save_model(model, epoch, models_dir / y_city[0].name)


def main(epochs, models_dir, train_path: Path, mask_dir: Path):
    cities = extract_ims(train_path)
    for i, (train_cities, val_city) in enumerate(KFold(n_splits=4, shuffle=False) \
            .split(cities)):
        if i < 2:
            continue
        print(f'Validation on {cities[val_city][0].name}')
        train(epochs, models_dir, cities[train_cities], cities[val_city], mask_dir)


if __name__ == "__main__":
    main(15, MODELS_DIR, TRAIN_PATH, MASK_DIR)
