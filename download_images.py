from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead"
METADATA_FILES = [
    "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv",
    "nutrition5k_dataset_metadata_dish_metadata_cafe2.csv",
]


def load_dishes(limit: int = 1500) -> list[tuple[str, float]]:
    dishes = []
    for filename in METADATA_FILES:
        with (DATA_DIR / filename).open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                dishes.append((parts[0], float(parts[1])))
                if len(dishes) >= limit:
                    return dishes
    return dishes


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_images(dishes: list[tuple[str, float]]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    total = len(dishes)
    downloaded = len(list(IMAGE_DIR.glob("*.png")))

    print(f"Target dishes: {total}")
    print(f"Existing images: {downloaded}")

    with requests.Session() as session:
        for index, (dish_id, _calories) in enumerate(dishes, start=1):
            save_path = IMAGE_DIR / f"{dish_id}.png"
            if save_path.exists() and save_path.stat().st_size > 0:
                continue

            url = f"{BASE_URL}/{dish_id}/rgb.png"
            try:
                response = session.get(url, timeout=20)
                if response.status_code == 200:
                    save_path.write_bytes(response.content)
                    print(f"{index}/{total} downloaded {dish_id}")
                else:
                    print(f"{index}/{total} missing {dish_id} ({response.status_code})")
            except requests.RequestException as exc:
                print(f"{index}/{total} skipped {dish_id}: {exc}")


def write_splits(dishes: list[tuple[str, float]]) -> None:
    dish_calories = dict(dishes)
    rows = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        dish_id = image_path.stem
        calories = dish_calories.get(dish_id)
        if calories is None or calories <= 0 or not is_valid_image(image_path):
            continue

        rows.append(
            {
                "dish_id": dish_id,
                "image_path": str(image_path.relative_to(DATA_DIR)),
                "calories": calories,
            }
        )

    if not rows:
        raise RuntimeError("No valid downloaded images were found.")

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Valid images: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")


def main() -> None:
    dishes = load_dishes()
    download_images(dishes)
    write_splits(dishes)


if __name__ == "__main__":
    main()
