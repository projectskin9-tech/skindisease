"""
app.py
======
Streamlit application for AI-Based Skin Disease Detection and
Affected-Area Severity Estimation.

Integrates:
    - Trained classification model (train_classifier.py)
    - Trained segmentation model (train_segmentation.py)
    - Affected-area percentage + severity band (area_severity.py)
    - Grad-CAM explainability (gradcam.py)

Usage:
    streamlit run app.py

Before running, edit the CONFIG section below to point to your trained
model files and class_indices.json.
"""

import json
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from PIL import Image

from gradcam import overlay_heatmap
from area_severity import compute_affected_area_percentage, severity_band

# ----------------------------------------------------------------------------
# CONFIG - update these paths to your trained WEIGHTS files
# (weights-only loading avoids Keras version mismatch problems between
#  Colab and your local machine)
# ----------------------------------------------------------------------------
CLASSIFIER_WEIGHTS_PATH = "outputs/mobilenet_classifier.weights.h5"
SEGMENTATION_WEIGHTS_PATH = "outputs_seg/unet_seg.weights.h5"
CLASS_INDICES_PATH = "outputs/class_indices.json"
CLASSIFIER_MODEL_NAME = "mobilenet"  # must match what was used in train_classifier.py
CLS_IMG_SIZE = (224, 224)
SEG_IMG_SIZE = (256, 256)
LOW_THRESHOLD = 10.0
HIGH_THRESHOLD = 30.0


# ----------------------------------------------------------------------------
# Architecture rebuild (must exactly match train_classifier.py / train_segmentation.py)
# ----------------------------------------------------------------------------
def _get_base_model(name, input_shape=(224, 224, 3)):
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


def _build_classifier_architecture(model_name, num_classes):
    base_model = _get_base_model(model_name)
    base_model.trainable = True  # matches fine-tuned state; weights overwrite anyway

    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def _build_unet_architecture(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    c1 = _conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = _conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = _conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    c4 = _conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    bn = _conv_block(p4, 512)

    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    d4 = _conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return models.Model(inputs, outputs, name="unet")


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    model = _build_classifier_architecture(CLASSIFIER_MODEL_NAME, num_classes)
    model.load_weights(CLASSIFIER_WEIGHTS_PATH)
    return model, idx_to_class


@st.cache_resource
def load_segmentation_model():
    model = _build_unet_architecture(input_shape=(*SEG_IMG_SIZE, 3))
    model.load_weights(SEGMENTATION_WEIGHTS_PATH)
    return model


def get_preprocess_fn(model_name):
    if model_name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError("Unknown model name")
    return preprocess_input


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------
def run_classification(pil_image, model, idx_to_class):
    preprocess_input = get_preprocess_fn(CLASSIFIER_MODEL_NAME)
    img_resized = pil_image.resize(CLS_IMG_SIZE)
    arr = np.expand_dims(np.array(img_resized).astype(np.float32), axis=0)
    arr_preprocessed = preprocess_input(arr.copy())

    preds = model.predict(arr_preprocessed)[0]
    pred_index = int(np.argmax(preds))
    predicted_class = idx_to_class[pred_index]
    confidence = float(preds[pred_index]) * 100
    return predicted_class, confidence, arr_preprocessed, preds


def run_segmentation(pil_image, seg_model):
    img_resized = pil_image.resize(SEG_IMG_SIZE)
    arr = np.expand_dims(np.array(img_resized).astype(np.float32) / 255.0, axis=0)
    mask_pred = seg_model.predict(arr)[0, ..., 0]  # (H, W)
    binary_mask = (mask_pred > 0.5).astype(np.uint8)
    return binary_mask, mask_pred


def run_gradcam(arr_preprocessed, model, original_pil_image, pred_index):
    """
    Manual layer-by-layer Grad-CAM.

    We avoid rebuilding a new Keras Functional model from an existing model's
    nested layers (model.get_layer(...).output) because Keras 3 has a known
    bug tracing graphs through nested sub-models (e.g. MobileNetV2 nested
    inside our classifier), which raises 'KeyError: tensor_dict[id(x)]'.

    Instead we call each layer directly under a GradientTape, in the exact
    order they were assembled in _build_classifier_architecture():
        model.layers = [Input, base_model, GAP, Dropout, Dense, Dropout, Dense]
    """
    img_tensor = tf.convert_to_tensor(arr_preprocessed, dtype=tf.float32)

    base_model = model.layers[1]
    gap_layer = model.layers[2]
    dropout1 = model.layers[3]
    dense1 = model.layers[4]
    dropout2 = model.layers[5]
    dense2 = model.layers[6]

    with tf.GradientTape() as tape:
        conv_output = base_model(img_tensor, training=False)
        tape.watch(conv_output)
        x = gap_layer(conv_output)
        x = dropout1(x, training=False)
        x = dense1(x)
        x = dropout2(x, training=False)
        predictions = dense2(x)

        if pred_index is None:
            pred_index_local = int(tf.argmax(predictions[0]))
        else:
            pred_index_local = pred_index
        class_channel = predictions[:, pred_index_local]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output_single = conv_output[0]
    heatmap = conv_output_single @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original_bgr = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
    overlay_bgr = overlay_heatmap(original_bgr, heatmap)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Skin Disease Detection", layout="wide")
    st.title("AI-Based Skin Disease Detection and Affected-Area Severity Estimation")
    st.caption(
        "Academic prototype. The affected-area percentage is an image-based "
        "estimate and is NOT a clinical severity score."
    )

    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Loading models..."):
        classifier, idx_to_class = load_classifier()
        seg_model = load_segmentation_model()

    with st.spinner("Running classification..."):
        predicted_class, confidence, arr_preprocessed, preds = run_classification(
            pil_image, classifier, idx_to_class
        )
        pred_index = int(np.argmax(preds))

    with st.spinner("Running segmentation..."):
        binary_mask, raw_mask = run_segmentation(pil_image, seg_model)
        affected_pct = compute_affected_area_percentage(binary_mask)
        band = severity_band(affected_pct, LOW_THRESHOLD, HIGH_THRESHOLD)

    with st.spinner("Generating Grad-CAM explanation..."):
        gradcam_overlay = run_gradcam(arr_preprocessed, classifier, pil_image, pred_index)

    with col2:
        st.subheader("Segmentation Mask")
        st.image(binary_mask * 255, use_container_width=True, clamp=True)

    st.divider()
    st.subheader("Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Condition", predicted_class)
    r2.metric("Model Confidence", f"{confidence:.1f}%")
    r3.metric("Visible Affected Area", f"{affected_pct:.1f}%")

    band_color = {"Low": "green", "Moderate": "orange", "High": "red"}[band]
    st.markdown(f"**Project-Defined Severity Band:** :{band_color}[{band}]")

    st.subheader("Grad-CAM Explanation")
    st.image(gradcam_overlay, use_container_width=True,
              caption="Heatmap of regions influencing the classification prediction")

    st.divider()
    st.warning(
        "⚠️ Disclaimer: This tool is an academic and decision-support prototype only. "
        "The affected-area percentage reflects the visible region in the supplied "
        "image and must not be interpreted as a clinical severity score or a "
        "substitute for professional medical assessment. Consult a qualified "
        "healthcare provider for diagnosis and treatment."
    )


if __name__ == "__main__":
    main()