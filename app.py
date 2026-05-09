from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    models = None
    transforms = None
    TORCH_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
SPLIT_IMAGE_DIR = DATA_DIR / "split_images"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.json"

MODEL_OPTIONS = {
    "CNN from scratch": {
        "weights": MODELS_DIR / "cnn_from_scratch.pth",
        "full": MODELS_DIR / "cnn_from_scratch_full.pt",
    },
    "MobileNetV2 transfer learning": {
        "weights": MODELS_DIR / "mobilenet_v2_finetuned.pth",
        "full": MODELS_DIR / "mobilenet_v2_full.pt",
    },
}


st.set_page_config(
    page_title="Food Calorie Estimator",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        df = load_csv(DATA_DIR / f"{split}.csv")
        if not df.empty:
            df = df.copy()
            df["split"] = split
            df["filename"] = df["image_path"].apply(lambda value: Path(value).name)
            df["absolute_path"] = df["image_path"].apply(lambda value: DATA_DIR / value)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def metric_text(value: object, decimals: int = 2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def image_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def confidence_from_mae(prediction: float, model_name: str, comparison: pd.DataFrame) -> Optional[float]:
    if comparison.empty or "MAE" not in comparison.columns:
        return None
    row = comparison[comparison["model"] == model_name]
    if row.empty:
        return None
    mae = float(row.iloc[0]["MAE"])
    return float(np.clip(100 * (1 - mae / max(abs(prediction), mae, 1)), 0, 100))


def dataset_match(filename: Optional[str], dataset: pd.DataFrame) -> pd.DataFrame:
    if not filename or dataset.empty:
        return pd.DataFrame()
    return dataset[dataset["filename"] == Path(filename).name]


def load_target_scaler() -> Optional[dict[str, float]]:
    if not TARGET_SCALER_PATH.exists():
        return None
    try:
        return json.loads(TARGET_SCALER_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def inverse_scaled_prediction(value: float) -> float:
    scaler = load_target_scaler()
    if not scaler:
        return value
    return value * float(scaler["std"]) + float(scaler["mean"])


if torch is not None:
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


    def build_mobilenet_v2_regressor():
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )
        return model


@st.cache_resource
def load_model(model_name: str):
    if torch is None:
        return None, f"PyTorch could not be loaded: {TORCH_IMPORT_ERROR}"

    paths = MODEL_OPTIONS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not paths["weights"].exists():
        if paths["full"].exists():
            model = torch.load(paths["full"], map_location=device, weights_only=False)
            model.eval()
            return model.to(device), None
        return None, f"No saved weights found for {model_name}."

    if model_name == "CNN from scratch":
        model = ScratchCNN()
    else:
        model = build_mobilenet_v2_regressor()

    try:
        model.load_state_dict(torch.load(paths["weights"], map_location=device))
    except RuntimeError as exc:
        return None, (
            f"Saved weights for {model_name} do not match the current improved architecture. "
            "Run train_models.py again to regenerate the model files. "
            f"Details: {exc}"
        )
    model.eval()
    return model.to(device), None


def predict_calories(image: Image.Image, model_name: str, comparison: pd.DataFrame):
    model, error = load_model(model_name)
    if error:
        return None, None, error

    device = next(model.parameters()).device
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_prediction = float(model(tensor).cpu().numpy().reshape(-1)[0])
        prediction = inverse_scaled_prediction(raw_prediction)

    confidence = confidence_from_mae(prediction, model_name, comparison)
    return prediction, confidence, None


comparison = load_csv(RESULTS_DIR / "model_comparison.csv")
cnn_history = load_csv(RESULTS_DIR / "cnn_training_history.csv")
mobilenet_history = load_csv(RESULTS_DIR / "mobilenet_v2_training_history.csv")
dataset = load_dataset()

st.title("Food Calorie Estimator")
st.caption("Nutrition5k image calorie estimation project")

overview_tab, results_tab, data_tab, prediction_tab = st.tabs(
    ["Overview", "Metrics and Curves", "Dataset", "Predict Calories"]
)

with overview_tab:
    st.subheader("Project Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Downloaded Images", image_count(IMAGE_DIR))
    col2.metric("Train Images", image_count(SPLIT_IMAGE_DIR / "train"))
    col3.metric("Validation Images", image_count(SPLIT_IMAGE_DIR / "val"))
    col4.metric("Test Images", image_count(SPLIT_IMAGE_DIR / "test"))

    if not comparison.empty:
        best_model = comparison.sort_values("MAE", ascending=True).iloc[0]
        st.divider()
        best_cols = st.columns(5)
        best_cols[0].metric("Best Model", best_model["model"])
        best_cols[1].metric("MAE", metric_text(best_model["MAE"]))
        best_cols[2].metric("MSE", metric_text(best_model["MSE"]))
        best_cols[3].metric("RMSE", metric_text(best_model["RMSE"]))
        best_cols[4].metric("R2 Score", metric_text(best_model["R2"], 3))

    model_files = sorted(path.name for path in MODELS_DIR.glob("*") if path.name != ".gitkeep")
    st.subheader("Saved Model Files")
    if model_files:
        st.write(model_files)
    else:
        st.warning("No trained model files were found in the models folder yet.")

with results_tab:
    st.subheader("Comparison Table")
    if comparison.empty:
        st.info("No model comparison CSV found.")
    else:
        st.dataframe(
            comparison.style.format(
                {
                    "trainable_parameters": "{:,.0f}",
                    "training_time_seconds": "{:,.2f}",
                    "MAE": "{:,.2f}",
                    "MSE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "R2": "{:,.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Error Metrics")
            st.bar_chart(comparison.set_index("model")[["MAE", "RMSE"]], use_container_width=True)
        with chart_col2:
            st.caption("R2 Score")
            st.bar_chart(comparison.set_index("model")[["R2"]], use_container_width=True)

    st.divider()
    st.subheader("Loss Curves")
    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.caption("CNN from scratch")
        if cnn_history.empty:
            st.info("No CNN training history found.")
        else:
            st.line_chart(cnn_history.set_index("epoch")[["train_loss", "val_loss"]])
            st.dataframe(cnn_history, use_container_width=True, hide_index=True)

    with curve_col2:
        st.caption("MobileNetV2 transfer learning")
        if mobilenet_history.empty:
            st.info("No MobileNetV2 training history found.")
        else:
            display_history = mobilenet_history.copy()
            display_history["step"] = range(1, len(display_history) + 1)
            st.line_chart(display_history.set_index("step")[["train_loss", "val_loss"]])
            st.dataframe(display_history, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Dataset Splits")
    if dataset.empty:
        st.info("No train, validation, or test CSV files found.")
    else:
        split_counts = dataset["split"].value_counts().rename_axis("split").reset_index(name="images")
        st.dataframe(split_counts, use_container_width=True, hide_index=True)

        selected_split = st.selectbox("Preview split", ["train", "val", "test"])
        split_df = dataset[dataset["split"] == selected_split].head(12)
        cols = st.columns(4)
        for index, row in enumerate(split_df.itertuples(index=False)):
            with cols[index % 4]:
                image_path = DATA_DIR / row.image_path
                if image_path.exists():
                    st.image(str(image_path), caption=f"{row.dish_id} | {row.calories:.1f} cal")

        st.subheader("Split CSV")
        st.dataframe(
            dataset[["split", "dish_id", "image_path", "calories"]],
            use_container_width=True,
            hide_index=True,
        )

with prediction_tab:
    st.subheader("Upload or Select a Food Image")

    source = st.radio("Image source", ["Upload image", "Choose from train images"], horizontal=True)
    image = None
    selected_filename = None

    if source == "Upload image":
        uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            selected_filename = uploaded.name
            image = Image.open(uploaded).convert("RGB")
    else:
        train_df = dataset[dataset["split"] == "train"].copy() if not dataset.empty else pd.DataFrame()
        if train_df.empty:
            st.info("No train split images found.")
        else:
            labels = train_df["filename"].tolist()
            selected_filename = st.selectbox("Train image", labels)
            row = train_df[train_df["filename"] == selected_filename].iloc[0]
            image = Image.open(DATA_DIR / row["image_path"]).convert("RGB")

    if image is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption=selected_filename, use_container_width=True)

        with right:
            model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
            if st.button("Estimate Calories", type="primary"):
                prediction, confidence, error = predict_calories(image, model_name, comparison)
                if error:
                    match = dataset_match(selected_filename, dataset)
                    if not match.empty:
                        row = match.iloc[0]
                        st.metric("Dataset Calories", metric_text(row["calories"]))
                        st.metric("Source", "Known dataset label")
                        st.caption(
                            "This image already exists in the prepared dataset, so the app "
                            "is showing its known label because model inference is currently "
                            f"unavailable: {error}"
                        )
                    else:
                        st.error(error)
                        st.info(
                            "Train the notebook first so it creates files like "
                            "models/cnn_from_scratch.pth or models/mobilenet_v2_finetuned.pth."
                        )
                else:
                    st.metric("Predicted Calories", metric_text(prediction))
                    st.metric(
                        "Confidence Estimate",
                        "N/A" if confidence is None else f"{confidence:.1f}%",
                    )

            if selected_filename and not dataset.empty:
                match = dataset_match(selected_filename, dataset)
                if not match.empty:
                    row = match.iloc[0]
                    st.divider()
                    st.caption("Dataset label for this known image")
                    st.metric("Actual Calories", metric_text(row["calories"]))
                    st.write(f"Split: {row['split']}")
