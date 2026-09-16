"""
train_classifier.py
===================
Improved HAM10000 7-class skin-lesion classifier.

Main improvements over the previous version:
    - ImageNet-pretrained MobileNetV2
    - Stronger but realistic augmentation
    - Balanced class weighting using sqrt-inverse frequency
    - Two-stage training
    - BatchNorm kept in inference mode during fine-tuning
    - Model selection based on validation macro-F1
    - Final test classification report + confusion matrix
    - Saves weights in the exact format expected by app.py

Expected:
    dataset/
        train/{akiec,bcc,bkl,df,mel,nv,vasc}/...
        val/{akiec,bcc,bkl,df,mel,nv,vasc}/...
        test/{akiec,bcc,bkl,df,mel,nv,vasc}/...

Usage in Colab:
    python train_classifier.py \
        --data_dir dataset \
        --model mobilenet \
        --epochs 25 \
        --fine_tune_epochs 15
"""

import os
import argparse
import json
from datetime import datetime
from collections import Counter

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================================
# CONFIG
# ============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EXPECTED_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


# ============================================================================
# MODEL
# ============================================================================

def get_base_model(name: str, input_shape=(224, 224, 3)):
    """Return ImageNet-pretrained backbone and preprocessing function."""

    name = name.lower()

    if name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        base = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    elif name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        base = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    elif name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input

        base = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    else:
        raise ValueError(
            f"Unknown model: {name}. "
            "Choose mobilenet, resnet50, or efficientnet."
        )

    return base, preprocess_input


def build_model(base_model, num_classes: int):
    """
    Same classifier head expected by the current Streamlit app:

        MobileNetV2
        GlobalAveragePooling2D
        Dropout
        Dense(128)
        Dropout
        Dense(num_classes)
    """

    # Start with the entire backbone frozen.
    base_model.trainable = False

    inputs = layers.Input(
        shape=base_model.input_shape[1:]
    )

    # Keep BatchNorm layers in inference mode.
    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.35
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.25
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return models.Model(
        inputs,
        outputs
    )


# ============================================================================
# DATA
# ============================================================================

def build_data_generators(
    data_dir: str,
    preprocess_fn
):
    """Create augmented train generator and clean validation/test generators."""

    train_dir = os.path.join(
        data_dir,
        "train"
    )

    val_dir = os.path.join(
        data_dir,
        "val"
    )

    test_dir = os.path.join(
        data_dir,
        "test"
    )

    # Stronger augmentation for training.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,

        rotation_range=25,
        width_shift_range=0.12,
        height_shift_range=0.12,

        shear_range=0.10,
        zoom_range=0.20,

        horizontal_flip=True,
        vertical_flip=True,

        brightness_range=[
            0.80,
            1.20
        ],

        fill_mode="reflect",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = None

    if os.path.isdir(test_dir):

        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )

    return train_gen, val_gen, test_gen


# ============================================================================
# BALANCED CLASS WEIGHTS
# ============================================================================

def compute_balanced_class_weights(train_gen):
    """
    Use sqrt-inverse frequency rather than raw inverse frequency.

    Raw inverse weighting can become extremely aggressive for df/vasc.
    Sqrt weighting still gives minority classes extra importance while
    reducing the chance of severe over-correction.
    """

    counts = Counter(
        train_gen.classes
    )

    num_classes = train_gen.num_classes
    total = sum(
        counts.values()
    )

    raw_weights = {}

    for cls_idx in range(num_classes):

        count = counts.get(
            cls_idx,
            1
        )

        raw_weights[cls_idx] = (
            total /
            (num_classes * count)
        )

    # Square-root the inverse-frequency weights.
    sqrt_weights = {
        idx: np.sqrt(weight)
        for idx, weight in raw_weights.items()
    }

    # Normalize around mean weight = 1.
    mean_weight = np.mean(
        list(sqrt_weights.values())
    )

    class_weight = {
        idx: float(weight / mean_weight)
        for idx, weight in sqrt_weights.items()
    }

    print("\nTraining class counts:")

    for idx in sorted(counts):

        class_name = None

        for name, value in train_gen.class_indices.items():

            if value == idx:
                class_name = name
                break

        print(
            f"  {idx}: "
            f"{class_name:<6} "
            f"{counts[idx]:>5} images "
            f"-> weight {class_weight[idx]:.3f}"
        )

    return class_weight


# ============================================================================
# MACRO-F1 CALLBACK
# ============================================================================

class MacroF1Callback(Callback):
    """
    Computes validation macro-F1 after every epoch.

    The best classifier is selected using macro-F1 rather than accuracy,
    because accuracy can be dominated by the very large nv class.
    """

    def __init__(
        self,
        validation_generator,
        save_path
    ):

        super().__init__()

        self.validation_generator = (
            validation_generator
        )

        self.save_path = save_path

        self.best_f1 = -np.inf

        self.history = []

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        self.validation_generator.reset()

        probabilities = self.model.predict(
            self.validation_generator,
            verbose=0
        )

        y_pred = np.argmax(
            probabilities,
            axis=1
        )

        y_true = self.validation_generator.classes

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        self.history.append(
            float(macro_f1)
        )

        print(
            f"\nValidation Macro-F1: "
            f"{macro_f1:.4f}"
        )

        if macro_f1 > self.best_f1:

            self.best_f1 = macro_f1

            self.model.save_weights(
                self.save_path
            )

            print(
                "Saved new best model "
                f"(val_macro_f1={macro_f1:.4f})"
            )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model,
    generator,
    class_indices,
    output_dir,
    split_name
):

    if generator is None:
        return

    generator.reset()

    probabilities = model.predict(
        generator,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    y_true = generator.classes

    # Reverse mapping.
    idx_to_class = {
        value: key
        for key, value in class_indices.items()
    }

    class_names = [
        idx_to_class[i]
        for i in range(
            len(idx_to_class)
        )
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        )
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"{split_name.upper()} CLASSIFICATION REPORT"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        report_text
    )

    print(
        f"{split_name} accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"{split_name} macro-F1: "
        f"{macro_f1:.4f}"
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    metrics_path = os.path.join(
        output_dir,
        f"{split_name.lower()}_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    print(
        f"Saved: {metrics_path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset root containing train/val/test"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet",
        choices=[
            "mobilenet",
            "resnet50",
            "efficientnet"
        ]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------------

    np.random.seed(42)
    tf.random.set_seed(42)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------

    base_model, preprocess_fn = get_base_model(
        args.model
    )

    train_gen, val_gen, test_gen = (
        build_data_generators(
            args.data_dir,
            preprocess_fn
        )
    )

    num_classes = (
        train_gen.num_classes
    )

    class_indices = (
        train_gen.class_indices
    )

    print(
        "\nClasses found:"
    )

    print(
        class_indices
    )

    # Verify expected 7-class setup.
    found_classes = sorted(
        class_indices.keys()
    )

    print(
        "\nExpected classes:"
    )

    print(
        EXPECTED_CLASSES
    )

    print(
        "\nFound classes:"
    )

    print(
        found_classes
    )

    if set(found_classes) != set(
        EXPECTED_CLASSES
    ):

        raise ValueError(
            "Dataset classes do not match the expected HAM10000 "
            "7-class setup.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found: {found_classes}"
        )

    # Save class mapping.
    with open(
        os.path.join(
            args.output_dir,
            "class_indices.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_indices,
            f,
            indent=2
        )

    # ------------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------------

    class_weight = (
        compute_balanced_class_weights(
            train_gen
        )
    )

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------

    model = build_model(
        base_model,
        num_classes
    )

    # ------------------------------------------------------------------------
    # Stage 1: classification head
    # ------------------------------------------------------------------------

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=args.lr
        ),

        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),

        metrics=[
            "accuracy"
        ]
    )

    model.summary()

    best_weights_path = os.path.join(
        args.output_dir,
        f"{args.model}_classifier.weights.h5"
    )

    macro_f1_callback = MacroF1Callback(
        val_gen,
        best_weights_path
    )

    callbacks_stage1 = [

        macro_f1_callback,

        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=False,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 1: TRAINING CLASSIFICATION HEAD"
    )

    print(
        "=" * 70
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_stage1,
        class_weight=class_weight,
        verbose=1,
    )

    # ------------------------------------------------------------------------
    # Stage 2: fine-tuning
    # ------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 2: FINE-TUNING LAST 30% OF BACKBONE"
    )

    print(
        "=" * 70
    )

    base_layer = model.layers[1]

    base_layer.trainable = True

    fine_tune_at = int(
        len(base_layer.layers) * 0.70
    )

    for layer in base_layer.layers:

        layer.trainable = (
            layer.index >= fine_tune_at
        )

        # Keep BatchNorm frozen/inference-only.
        if isinstance(
            layer,
            layers.BatchNormalization
        ):

            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=args.fine_tune_lr
        ),

        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),

        metrics=[
            "accuracy"
        ]
    )

    # New callback for fine-tuning.
    macro_f1_callback_ft = MacroF1Callback(
        val_gen,
        best_weights_path
    )

    macro_f1_callback_ft.best_f1 = (
        macro_f1_callback.best_f1
    )

    callbacks_stage2 = [

        macro_f1_callback_ft,

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=False,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks_stage2,
        class_weight=class_weight,
        verbose=1,
    )

    # ------------------------------------------------------------------------
    # Load best macro-F1 weights
    # ------------------------------------------------------------------------

    print(
        "\nLoading best validation Macro-F1 weights..."
    )

    model.load_weights(
        best_weights_path
    )

    # ------------------------------------------------------------------------
    # Save final model weights
    # ------------------------------------------------------------------------

    final_weights_path = os.path.join(
        args.output_dir,
        f"{args.model}_classifier_final.weights.h5"
    )

    model.save_weights(
        final_weights_path
    )

    print(
        f"Saved final weights: "
        f"{final_weights_path}"
    )

    # ------------------------------------------------------------------------
    # Save complete model
    # ------------------------------------------------------------------------

    final_model_path = os.path.join(
        args.output_dir,
        f"{args.model}_classifier_final.keras"
    )

    model.save(
        final_model_path
    )

    print(
        f"Saved complete model: "
        f"{final_model_path}"
    )

    # ------------------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------------------

    history = {}

    for key, value in history1.history.items():

        history[
            f"stage1_{key}"
        ] = [
            float(v)
            for v in value
        ]

    for key, value in history2.history.items():

        history[
            f"stage2_{key}"
        ] = [
            float(v)
            for v in value
        ]

    history[
        "stage1_val_macro_f1"
    ] = macro_f1_callback.history

    history[
        "stage2_val_macro_f1"
    ] = macro_f1_callback_ft.history

    history_path = os.path.join(
        args.output_dir,
        "history.json"
    )

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # ------------------------------------------------------------------------
    # Evaluate validation
    # ------------------------------------------------------------------------

    evaluate_model(
        model,
        val_gen,
        class_indices,
        args.output_dir,
        "validation"
    )

    # ------------------------------------------------------------------------
    # Evaluate test
    # ------------------------------------------------------------------------

    evaluate_model(
        model,
        test_gen,
        class_indices,
        args.output_dir,
        "test"
    )

    # ------------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------------

    summary = {
        "model": args.model,
        "num_classes": num_classes,
        "classes": found_classes,
        "best_validation_macro_f1": float(
            max(
                macro_f1_callback.history
                + macro_f1_callback_ft.history
            )
        ),
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(
        args.output_dir,
        "training_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best validation Macro-F1: "
        f"{summary['best_validation_macro_f1']:.4f}"
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
