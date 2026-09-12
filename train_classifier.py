"""
Skin Disease Classification - Training Script
================================================
Trains a transfer-learning based classifier (MobileNetV2 / ResNet50 / EfficientNetB0)
on the HAM10000 dataset (or any similarly structured image classification dataset).

Expected dataset structure (after you organize HAM10000 into class folders):

    dataset/
        train/
            akiec/
            bcc/
            bkl/
            df/
            mel/
            nv/
            vasc/
        val/
            akiec/
            ...
        test/
            akiec/
            ...

If your HAM10000 download is a flat folder of images + a CSV of labels (the usual
Kaggle format), run `organize_ham10000.py` (ask me for it) first to sort images
into class folders like the structure above.

Usage:
    python train_classifier.py --data_dir dataset --model mobilenet --epochs 30
"""

import os
import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_base_model(name: str, input_shape=(224, 224, 3)):
    """Return (base_model, preprocess_fn) for the chosen architecture."""
    name = name.lower()
    if name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model: {name}. Choose mobilenet, resnet50, or efficientnet.")
    return base, preprocess_input


def build_model(base_model, num_classes: int, fine_tune: bool = False, fine_tune_at: int = None):
    """Attach a classification head on top of the frozen (or partially unfrozen) base."""
    base_model.trainable = fine_tune
    if fine_tune and fine_tune_at is not None:
        # Freeze all layers before `fine_tune_at`, unfreeze the rest
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def build_data_generators(data_dir: str, preprocess_fn):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (with train/ val/ test/)")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "resnet50", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Data
    base_model, preprocess_fn = get_base_model(args.model)
    train_gen, val_gen = build_data_generators(args.data_dir, preprocess_fn)
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    print(f"Classes found: {class_indices}")

    with open(os.path.join(args.output_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)

    # 2. Build model (head-only training first)
    model = build_model(base_model, num_classes, fine_tune=False)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = os.path.join(args.output_dir, f"{args.model}_best.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    print("\n=== Stage 1: Training classification head (base frozen) ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 3. Fine-tune: unfreeze top portion of base model
    print("\n=== Stage 2: Fine-tuning top layers of base model ===")
    base_layer = model.layers[1]  # the base_model sits as a nested layer
    base_layer.trainable = True
    fine_tune_at = int(len(base_layer.layers) * 0.7)  # unfreeze last 30%
    for layer in base_layer.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    # 4. Save final model
    final_path = os.path.join(args.output_dir, f"{args.model}_final.h5")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    # 5. Save training history
    hist = {**{k: v for k, v in history1.history.items()},
            **{f"ft_{k}": v for k, v in history2.history.items()}}
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    print(f"\nDone. Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
