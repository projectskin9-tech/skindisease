"""
organize_ham10000.py
=====================
HAM10000 (Kaggle) usually comes as:
    HAM10000_images_part_1/  *.jpg
    HAM10000_images_part_2/  *.jpg
    HAM10000_metadata.csv   (columns include: image_id, dx, ...)

This script sorts those images into class folders and splits them into
train / val / test, ready for `train_classifier.py`.

Usage:
    python organize_ham10000.py \
        --images_dirs HAM10000_images_part_1 HAM10000_images_part_2 \
        --metadata HAM10000_metadata.csv \
        --output_dir dataset \
        --val_split 0.15 --test_split 0.15
"""

import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def find_image_path(image_id, images_dirs):
    for d in images_dirs:
        candidate = os.path.join(d, image_id + ".jpg")
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dirs", nargs="+", required=True,
                         help="One or more folders containing the raw .jpg images")
    parser.add_argument("--metadata", type=str, required=True,
                         help="Path to HAM10000_metadata.csv")
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"Loaded metadata: {len(df)} rows")
    print("Class distribution:\n", df["dx"].value_counts())

    # Split: train vs temp (val+test), then temp -> val/test
    train_df, temp_df = train_test_split(
        df, test_size=(args.val_split + args.test_split),
        stratify=df["dx"], random_state=args.seed,
    )
    relative_test_size = args.test_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size,
        stratify=temp_df["dx"], random_state=args.seed,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name}: {len(split_df)} images")
        missing = 0
        for _, row in split_df.iterrows():
            image_id = row["image_id"]
            label = row["dx"]
            src = find_image_path(image_id, args.images_dirs)
            if src is None:
                missing += 1
                continue
            dest_dir = os.path.join(args.output_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, image_id + ".jpg")
            shutil.copyfile(src, dest)
        if missing:
            print(f"  Warning: {missing} images not found on disk and skipped.")

    print(f"\nDone. Dataset organized under: {args.output_dir}/")
    print("Structure: dataset/{train,val,test}/<class_name>/*.jpg")


if __name__ == "__main__":
    main()
