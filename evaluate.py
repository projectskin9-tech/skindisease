"""
evaluate.py
===========
Generates official evaluation metrics for your project report:
    - Classifier: accuracy, precision, recall, F1, confusion matrix
    - Segmentation: IoU, Dice coefficient

Run this AFTER training both models. It rebuilds the architectures and
loads the saved WEIGHTS (same approach as app.py) to avoid Keras version
mismatch issues.

Usage:
    python evaluate.py --data_dir dataset --seg_data_dir seg_dataset \
        --classifier_weights outputs/mobilenet_classifier.weights.h5 \
        --seg_weights outputs_seg/unet_seg.weights.h5 \
        --class_indices outputs/class_indices.json
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
SEG_IMG_SIZE = (256, 256)
BATCH_SIZE = 32


# ----------------------------------------------------------------------------
# Architecture rebuild (must match train_classifier.py / app.py exactly)
# ----------------------------------------------------------------------------
def get_base_model(name, input_shape=(224, 224, 3)):
    name = name.lower()
    if name == "mobilenet":
        base = MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    elif name == "resnet50":
        base = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model: {name}")
    return base


def build_classifier_architecture(model_name, num_classes):
    base_model = get_base_model(model_name)
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet_architecture(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, 512)
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4]); d4 = conv_block(u4, 256)
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d4)
    u3 = layers.Concatenate()([u3, c3]); d3 = conv_block(u3, 128)
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2]); d2 = conv_block(u2, 64)
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1]); d1 = conv_block(u1, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return models.Model(inputs, outputs, name="unet")


# ----------------------------------------------------------------------------
# Classifier evaluation
# ----------------------------------------------------------------------------
def evaluate_classifier(data_dir, weights_path, class_indices_path, model_name, output_dir):
    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATION")
    print("=" * 60)

    with open(class_indices_path) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_indices)

    model = build_classifier_architecture(model_name, num_classes)
    model.load_weights(weights_path)

    if model_name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    test_dir = os.path.join(data_dir, "test")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )

    print(f"Evaluating on {test_gen.samples} test images...")
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nPer-Class Report:\n", report)

    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Classifier")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    print(f"\nSaved confusion matrix plot to {cm_path}")

    # Save metrics as JSON
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    metrics_path = os.path.join(output_dir, "classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


# ----------------------------------------------------------------------------
# Segmentation evaluation
# ----------------------------------------------------------------------------
def dice_coef_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def load_image(path, size, channels=3):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, size)
    return img.numpy()


def evaluate_segmentation(seg_data_dir, weights_path, output_dir):
    print("\n" + "=" * 60)
    print("SEGMENTATION EVALUATION")
    print("=" * 60)

    model = build_unet_architecture(input_shape=(*SEG_IMG_SIZE, 3))
    model.load_weights(weights_path)

    val_img_dir = os.path.join(seg_data_dir, "val", "images")
    val_mask_dir = os.path.join(seg_data_dir, "val", "masks")

    filenames = sorted(os.listdir(val_img_dir))
    dice_scores, iou_scores = [], []

    print(f"Evaluating on {len(filenames)} validation images...")
    for fname in filenames:
        stem = os.path.splitext(fname)[0]
        mask_candidates = [m for m in os.listdir(val_mask_dir) if m.startswith(stem)]
        if not mask_candidates:
            continue

        img = load_image(os.path.join(val_img_dir, fname), SEG_IMG_SIZE, channels=3) / 255.0
        gt_mask = load_image(os.path.join(val_mask_dir, mask_candidates[0]), SEG_IMG_SIZE, channels=1) / 255.0
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        pred_mask = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        dice_scores.append(dice_coef_np(gt_mask, pred_mask))
        iou_scores.append(iou_np(gt_mask, pred_mask))

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    print(f"\nMean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU:              {mean_iou:.4f}")

    metrics = {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "num_images_evaluated": len(dice_scores),
    }
    metrics_path = os.path.join(output_dir, "segmentation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Classifier dataset root (with test/ folder)")
    parser.add_argument("--seg_data_dir", type=str, help="Segmentation dataset root (with val/images, val/masks)")
    parser.add_argument("--classifier_weights", type=str, help="Path to classifier .weights.h5 file")
    parser.add_argument("--seg_weights", type=str, help="Path to segmentation .weights.h5 file")
    parser.add_argument("--class_indices", type=str, help="Path to class_indices.json")
    parser.add_argument("--model_name", type=str, default="mobilenet",
                         choices=["mobilenet", "resnet50", "efficientnet"])
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_dir and args.classifier_weights and args.class_indices:
        evaluate_classifier(
            args.data_dir, args.classifier_weights, args.class_indices,
            args.model_name, args.output_dir,
        )
    else:
        print("Skipping classifier evaluation (missing --data_dir, --classifier_weights, or --class_indices)")

    if args.seg_data_dir and args.seg_weights:
        evaluate_segmentation(args.seg_data_dir, args.seg_weights, args.output_dir)
    else:
        print("Skipping segmentation evaluation (missing --seg_data_dir or --seg_weights)")

    print(f"\nAll results saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()