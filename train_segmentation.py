"""
train_segmentation.py
======================
Trains a U-Net model to segment the affected skin region (lesion) from an image,
producing a binary mask used later for affected-area percentage calculation.

Expected dataset structure:

    seg_dataset/
        train/
            images/   *.jpg or *.png
            masks/    *.png   (same filename as corresponding image; binary mask)
        val/
            images/
            masks/

Masks should be single-channel images where affected pixels = 255 (or 1) and
background = 0. Datasets like ISIC Archive provide lesion segmentation masks
in this style.

Usage:
    python train_segmentation.py --data_dir seg_dataset --epochs 50
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (256, 256)
BATCH_SIZE = 16


# ----------------------------------------------------------------------------
# U-Net architecture
# ----------------------------------------------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    d4 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)

    return models.Model(inputs, outputs, name="unet")


# ----------------------------------------------------------------------------
# Loss / metrics
# ----------------------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_image_mask_paths(split_dir):
    img_dir = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    filenames = sorted(os.listdir(img_dir))
    image_paths, mask_paths = [], []
    for fname in filenames:
        stem = os.path.splitext(fname)[0]
        mask_candidates = [f for f in os.listdir(mask_dir) if f.startswith(stem)]
        if not mask_candidates:
            continue
        image_paths.append(os.path.join(img_dir, fname))
        mask_paths.append(os.path.join(mask_dir, mask_candidates[0]))
    return image_paths, mask_paths


def decode_image(path, channels=3, size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, size)
    return img


def load_pair(image_path, mask_path):
    img = decode_image(image_path, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    mask = decode_image(mask_path, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)  # binarize
    return img, mask


def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, mask


def build_dataset(image_paths, mask_paths, training=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(200)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path with train/ and val/ subfolders, each containing images/ and masks/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs_seg")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_imgs, train_masks = load_image_mask_paths(os.path.join(args.data_dir, "train"))
    val_imgs, val_masks = load_image_mask_paths(os.path.join(args.data_dir, "val"))
    print(f"Train pairs: {len(train_imgs)} | Val pairs: {len(val_imgs)}")

    train_ds = build_dataset(train_imgs, train_masks, training=True)
    val_ds = build_dataset(val_imgs, val_masks, training=False)

    model = build_unet(input_shape=(*IMG_SIZE, 3))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_metric],
    )
    model.summary()

    ckpt_path = os.path.join(args.output_dir, "unet_best.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    final_path = os.path.join(args.output_dir, "unet_final.h5")
    model.save(final_path)
    print(f"Saved final segmentation model to {final_path}")


if __name__ == "__main__":
    main()
