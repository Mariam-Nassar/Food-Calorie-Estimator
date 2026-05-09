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

This project predicts a continuous calorie value, so it is a regression task. Use MAE, RMSE, and R2 to judge model quality instead of classification accuracy.

You can also train from the command line:

```bash
python download_images.py
python train_models.py
```

For a quick smoke test, reduce epochs with environment variables:

```bash
CNN_EPOCHS=1 TRANSFER_HEAD_EPOCHS=1 TRANSFER_FINE_TUNE_EPOCHS=1 python train_models.py
```

## Current Results

The latest test comparison is saved in `results/model_comparison.csv`.
