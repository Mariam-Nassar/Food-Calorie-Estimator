import time
import os
import json
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

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
CNN_EPOCHS = int(os.getenv("CNN_EPOCHS", "25"))
TRANSFER_HEAD_EPOCHS = int(os.getenv("TRANSFER_HEAD_EPOCHS", "10"))
TRANSFER_FINE_TUNE_EPOCHS = int(os.getenv("TRANSFER_FINE_TUNE_EPOCHS", "15"))
SEED = 42
WEIGHT_DECAY = 1e-4
REUSE_EXISTING_WEIGHTS = os.getenv("REUSE_EXISTING_WEIGHTS", "0") == "1"


class TargetScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 0 else 1.0

    @classmethod
    def from_series(cls, values: pd.Series):
        return cls(values.mean(), values.std())

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.std

    def inverse_array(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2), encoding="utf-8")


class NutritionDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None, target_scaler: TargetScaler | None = None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.target_scaler = target_scaler
        self.data["resolved_image_path"] = self.data["image_path"].apply(self.resolve_image_path)
        before = len(self.data)
        self.data = self.data[self.data["resolved_image_path"].apply(lambda path: path.exists())].reset_index(drop=True)
        skipped = before - len(self.data)
        if skipped:
            print(f"Skipped {skipped} rows from {csv_path.name} because the image file was missing.")
        if self.data.empty:
            raise RuntimeError(f"No usable images found for {csv_path}.")

    @staticmethod
    def resolve_image_path(value: str) -> Path:
        image_path = Path(value)
        if image_path.is_absolute():
            return image_path
        candidate = DATA_DIR / image_path
        if candidate.exists():
            return candidate
        return DATA_DIR / "images" / image_path.name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = row["resolved_image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        calories = float(row["calories"])
        if self.target_scaler:
            calories = self.target_scaler.transform(calories)
        calories = torch.tensor([calories], dtype=torch.float32)
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.regressor(x)


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


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path, device, patience=6):
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d}/{epochs} train loss={train_loss:.4f} val loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    return pd.DataFrame(history), time.time() - start


def predict(model, loader, device, target_scaler: TargetScaler):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, y in tqdm(loader, leave=False):
            outputs = model(images.to(device)).cpu().numpy().reshape(-1)
            predictions.extend(outputs.tolist())
            targets.extend(y.numpy().reshape(-1).tolist())
    targets = target_scaler.inverse_array(np.array(targets))
    predictions = target_scaler.inverse_array(np.array(predictions))
    return targets, predictions


def evaluate_regression(model, loader, device, target_scaler: TargetScaler):
    y_true, y_pred = predict(model, loader, device, target_scaler)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": float(r2_score(y_true, y_pred))}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15),
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

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    target_scaler = TargetScaler.from_series(train_df["calories"])
    target_scaler.save(MODEL_DIR / "target_scaler.json")
    print(f"Target scaler: mean={target_scaler.mean:.2f}, std={target_scaler.std:.2f}")

    train_loader = DataLoader(
        NutritionDataset(DATA_DIR / "train.csv", train_transforms, target_scaler),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        NutritionDataset(DATA_DIR / "val.csv", eval_transforms, target_scaler),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        NutritionDataset(DATA_DIR / "test.csv", eval_transforms, target_scaler),
        batch_size=BATCH_SIZE,
    )

    criterion = nn.SmoothL1Loss(beta=0.5)

    scratch_cnn = ScratchCNN().to(device)
    cnn_weights = MODEL_DIR / "cnn_from_scratch.pth"
    if REUSE_EXISTING_WEIGHTS and cnn_weights.exists() and (RESULTS_DIR / "cnn_training_history.csv").exists():
        try:
            print("Using existing CNN weights.")
            scratch_cnn.load_state_dict(torch.load(cnn_weights, map_location=device))
            cnn_time = 0.0
        except RuntimeError:
            print("Existing CNN weights are incompatible with the improved architecture. Retraining CNN.")
            cnn_history, cnn_time = train_model(
                scratch_cnn,
                train_loader,
                val_loader,
                criterion,
                optim.AdamW(scratch_cnn.parameters(), lr=8e-4, weight_decay=WEIGHT_DECAY),
                CNN_EPOCHS,
                cnn_weights,
                device,
            )
            cnn_history.to_csv(RESULTS_DIR / "cnn_training_history.csv", index=False)
    else:
        cnn_history, cnn_time = train_model(
            scratch_cnn,
            train_loader,
            val_loader,
            criterion,
            optim.AdamW(scratch_cnn.parameters(), lr=8e-4, weight_decay=WEIGHT_DECAY),
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
        optim.AdamW(filter(lambda p: p.requires_grad, mobilenet_v2.parameters()), lr=8e-4, weight_decay=WEIGHT_DECAY),
        TRANSFER_HEAD_EPOCHS,
        MODEL_DIR / "mobilenet_v2_head_best.pth",
        device,
    )

    mobilenet_v2.load_state_dict(torch.load(MODEL_DIR / "mobilenet_v2_head_best.pth", map_location=device))
    for param in mobilenet_v2.features[-7:].parameters():
        param.requires_grad = True
    fine_history, fine_time = train_model(
        mobilenet_v2,
        train_loader,
        val_loader,
        criterion,
        optim.AdamW(filter(lambda p: p.requires_grad, mobilenet_v2.parameters()), lr=2e-5, weight_decay=WEIGHT_DECAY),
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
                **evaluate_regression(scratch_cnn, test_loader, device, target_scaler),
            },
            {
                "model": "MobileNetV2 transfer learning",
                "trainable_parameters": count_trainable_parameters(mobilenet_v2),
                "training_time_seconds": round(head_time + fine_time, 2),
                **evaluate_regression(mobilenet_v2, test_loader, device, target_scaler),
            },
        ]
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    torch.save(scratch_cnn, MODEL_DIR / "cnn_from_scratch_full.pt")
    torch.save(mobilenet_v2, MODEL_DIR / "mobilenet_v2_full.pt")
    print(comparison)


if __name__ == "__main__":
    main()
