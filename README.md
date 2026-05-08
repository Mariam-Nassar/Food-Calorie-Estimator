# Food-Calorie-Estimator
Deep learning project to estimate food calories from images using Nutrition5k dataset.

## Project Flow

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run `notebooks/01_data_preparation.ipynb` to download images, clean the data, resize/normalize images, and create the train/validation/test split.

3. Run `notebooks/02_model_training.ipynb` to train:

- CNN from scratch
- MobileNetV2 transfer learning model with a custom regression head

The training notebook saves generated model files under `models/` and result CSV files under `results/`.

## Current Results

The latest test comparison is saved in `results/model_comparison.csv`.
