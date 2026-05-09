import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TORCH_CACHE_DIR = PROJECT_ROOT / ".torch_cache"
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

BATCH_SIZE = 16
CNN_EPOCHS = 10
TRANSFER_HEAD_EPOCHS = 5
TRANSFER_FINE_TUNE_EPOCHS = 5
SEED = 42


class NutritionDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = DATA_DIR / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        calories = torch.tensor([float(row["calories"])], dtype=torch.float32)
        return image, calories


class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def build_mobilenet_v2_regressor(freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = not freeze_backbone
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)
            if optimizer:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path, device):
    history = []
    best_val_loss = float("inf")
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_one_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d}/{epochs} train MSE={train_loss:.2f} val MSE={val_loss:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved {save_path}")
    return pd.DataFrame(history), time.time() - start


def predict(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, y in tqdm(loader, leave=False):
            outputs = model(images.to(device)).cpu().numpy().reshape(-1)
            predictions.extend(outputs.tolist())
            targets.extend(y.numpy().reshape(-1).tolist())
    return np.array(targets), np.array(predictions)


def evaluate_regression(model, loader, device):
    y_true, y_pred = predict(model, loader, device)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": float(r2_score(y_true, y_pred))}


def main():
    torch.manual_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(NutritionDataset(DATA_DIR / "train.csv", train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NutritionDataset(DATA_DIR / "val.csv", eval_transforms), batch_size=BATCH_SIZE)
    test_loader = DataLoader(NutritionDataset(DATA_DIR / "test.csv", eval_transforms), batch_size=BATCH_SIZE)

    criterion = nn.MSELoss()

    scratch_cnn = ScratchCNN().to(device)
    cnn_weights = MODEL_DIR / "cnn_from_scratch.pth"
    if cnn_weights.exists() and (RESULTS_DIR / "cnn_training_history.csv").exists():
        print("Using existing CNN weights.")
        scratch_cnn.load_state_dict(torch.load(cnn_weights, map_location=device))
        cnn_time = 0.0
    else:
        cnn_history, cnn_time = train_model(
            scratch_cnn,
            train_loader,
            val_loader,
            criterion,
            optim.Adam(scratch_cnn.parameters(), lr=1e-3),
            CNN_EPOCHS,
            cnn_weights,
            device,
        )
        cnn_history.to_csv(RESULTS_DIR / "cnn_training_history.csv", index=False)

    mobilenet_v2 = build_mobilenet_v2_regressor(freeze_backbone=True).to(device)
    head_history, head_time = train_model(
        mobilenet_v2,
        train_loader,
        val_loader,
        criterion,
        optim.Adam(filter(lambda p: p.requires_grad, mobilenet_v2.parameters()), lr=1e-3),
        TRANSFER_HEAD_EPOCHS,
        MODEL_DIR / "mobilenet_v2_head_best.pth",
        device,
    )

    for param in mobilenet_v2.features[-4:].parameters():
        param.requires_grad = True
    fine_history, fine_time = train_model(
        mobilenet_v2,
        train_loader,
        val_loader,
        criterion,
        optim.Adam(filter(lambda p: p.requires_grad, mobilenet_v2.parameters()), lr=1e-5),
        TRANSFER_FINE_TUNE_EPOCHS,
        MODEL_DIR / "mobilenet_v2_finetuned.pth",
        device,
    )
    pd.concat(
        [
            head_history.assign(phase="frozen_backbone"),
            fine_history.assign(phase="fine_tuning"),
        ],
        ignore_index=True,
    ).to_csv(RESULTS_DIR / "mobilenet_v2_training_history.csv", index=False)

    scratch_cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_from_scratch.pth", map_location=device))
    mobilenet_v2.load_state_dict(torch.load(MODEL_DIR / "mobilenet_v2_finetuned.pth", map_location=device))

    comparison = pd.DataFrame(
        [
            {
                "model": "CNN from scratch",
                "trainable_parameters": count_trainable_parameters(scratch_cnn),
                "training_time_seconds": round(cnn_time, 2),
                **evaluate_regression(scratch_cnn, test_loader, device),
            },
            {
                "model": "MobileNetV2 transfer learning",
                "trainable_parameters": count_trainable_parameters(mobilenet_v2),
                "training_time_seconds": round(head_time + fine_time, 2),
                **evaluate_regression(mobilenet_v2, test_loader, device),
            },
        ]
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    torch.save(scratch_cnn, MODEL_DIR / "cnn_from_scratch_full.pt")
    torch.save(mobilenet_v2, MODEL_DIR / "mobilenet_v2_full.pt")
    print(comparison)


if __name__ == "__main__":
    main()
