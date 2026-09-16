"""
app.py
======
Streamlit application for:
- HAM10000 7-class skin lesion classification
- U-Net lesion segmentation
- Affected-area percentage
- Project-defined affected-area level
- Cancer-status grouping
- Grad-CAM explanation

Run:
    streamlit run app.py

IMPORTANT:
The classifier output is an academic model prediction, not a medical diagnosis.
The cancer-status grouping is a broad dataset/class grouping.
The affected-area level is an image-based project indicator, NOT clinical
disease severity.
"""

import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

from gradcam import overlay_heatmap
from area_severity import (
    compute_affected_area_percentage,
    severity_band,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Corrected EfficientNet 7-class model
CLASSIFIER_MODEL_PATH = "skin_classifier_fixed.keras"

# Existing U-Net segmentation weights
SEGMENTATION_WEIGHTS_PATH = "outputs_seg/unet_seg.weights.h5"

# Confirmed classifier input size
CLS_IMG_SIZE = (224, 224)

# Existing U-Net input size
SEG_IMG_SIZE = (256, 256)

# Project-defined affected-area thresholds
LOW_THRESHOLD = 10.0
HIGH_THRESHOLD = 30.0


# ============================================================================
# HAM10000 CLASS INFORMATION
# ============================================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


# Broad status grouping for display.
# akiec is treated separately because it is generally considered
# precancerous/premalignant rather than an established cancer diagnosis.
CANCER_STATUS = {
    "mel": "Cancerous / Malignant class",
    "bcc": "Cancerous / Malignant class",
    "akiec": "Pre-cancerous / Premalignant class",
    "bkl": "Non-cancerous / Benign class",
    "df": "Non-cancerous / Benign class",
    "nv": "Non-cancerous / Benign class",
    "vasc": "Non-cancerous class",
}


def get_cancer_status(predicted_class: str) -> str:
    """Return the broad display grouping for the predicted HAM10000 class."""
    return CANCER_STATUS.get(
        predicted_class,
        "Unknown classification group",
    )


# ============================================================================
# U-NET ARCHITECTURE
# ============================================================================

def _conv_block(x, filters):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    return x


def _build_unet_architecture(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = _conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = _conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = _conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = _conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = _conv_block(p4, 512)

    # Decoder
    u4 = layers.Conv2DTranspose(
        256,
        2,
        strides=2,
        padding="same",
    )(bn)

    u4 = layers.Concatenate()([u4, c4])
    d4 = _conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same",
    )(d4)

    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same",
    )(d3)

    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same",
    )(d2)

    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, 32)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
    )(d1)

    return models.Model(
        inputs,
        outputs,
        name="unet",
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_classifier():
    """Load the corrected EfficientNet 7-class classifier."""

    model = tf.keras.models.load_model(
        CLASSIFIER_MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != 7:
        raise ValueError(
            "Classifier must have exactly 7 outputs, "
            f"but found {model.output_shape[-1]}."
        )

    idx_to_class = {
        index: class_name
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return model, idx_to_class


@st.cache_resource
def load_segmentation_model():
    """Load the existing U-Net model from weights."""

    model = _build_unet_architecture(
        input_shape=(*SEG_IMG_SIZE, 3),
    )

    model.load_weights(
        SEGMENTATION_WEIGHTS_PATH,
    )

    return model


# ============================================================================
# CLASSIFICATION
# ============================================================================

def run_classification(
    pil_image,
    model,
    idx_to_class,
):
    """
    Run EfficientNet classification.

    The saved EfficientNet model uses 224x224 RGB images.
    tf.keras EfficientNet includes its input rescaling, so raw 0-255
    pixel values are passed to the model.
    """

    img_resized = pil_image.resize(
        CLS_IMG_SIZE,
    )

    arr = np.array(
        img_resized,
        dtype=np.float32,
    )

    arr = np.expand_dims(
        arr,
        axis=0,
    )

    # Keep raw 0-255 values.
    # EfficientNetB0 in tf.keras performs its own input rescaling.
    arr_preprocessed = arr

    preds = model.predict(
        arr_preprocessed,
        verbose=0,
    )[0]

    pred_index = int(
        np.argmax(preds)
    )

    predicted_class = idx_to_class[
        pred_index
    ]

    confidence = (
        float(preds[pred_index]) * 100.0
    )

    return (
        predicted_class,
        confidence,
        arr_preprocessed,
        preds,
    )


# ============================================================================
# SEGMENTATION
# ============================================================================

def run_segmentation(
    pil_image,
    seg_model,
):
    """Run U-Net segmentation."""

    img_resized = pil_image.resize(
        SEG_IMG_SIZE,
    )

    arr = np.array(
        img_resized,
        dtype=np.float32,
    )

    arr = arr / 255.0

    arr = np.expand_dims(
        arr,
        axis=0,
    )

    mask_pred = seg_model.predict(
        arr,
        verbose=0,
    )[0, ..., 0]

    binary_mask = (
        mask_pred > 0.5
    ).astype(np.uint8)

    return (
        binary_mask,
        mask_pred,
    )


# ============================================================================
# GRAD-CAM
# ============================================================================

def run_gradcam(
    arr_preprocessed,
    model,
    original_pil_image,
    pred_index,
):
    """
    Manual layer-by-layer Grad-CAM for the saved EfficientNet classifier.

    Expected outer model structure:
        Input
        EfficientNetB0
        GlobalAveragePooling2D
        Dropout
        Dense
        Dropout
        Dense
    """

    img_tensor = tf.convert_to_tensor(
        arr_preprocessed,
        dtype=tf.float32,
    )

    if len(model.layers) < 7:
        raise ValueError(
            "Unexpected classifier structure; "
            "Grad-CAM requires the trained EfficientNet classification head."
        )

    base_model = model.layers[1]
    gap_layer = model.layers[2]
    dropout1 = model.layers[3]
    dense1 = model.layers[4]
    dropout2 = model.layers[5]
    dense2 = model.layers[6]

    with tf.GradientTape() as tape:

        conv_output = base_model(
            img_tensor,
            training=False,
        )

        tape.watch(conv_output)

        x = gap_layer(
            conv_output,
        )

        x = dropout1(
            x,
            training=False,
        )

        x = dense1(x)

        x = dropout2(
            x,
            training=False,
        )

        predictions = dense2(x)

        class_channel = predictions[
            :,
            pred_index,
        ]

    grads = tape.gradient(
        class_channel,
        conv_output,
    )

    if grads is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    conv_output_single = conv_output[0]

    heatmap = (
        conv_output_single
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap,
    )

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    heatmap = heatmap.numpy()

    original_bgr = cv2.cvtColor(
        np.array(original_pil_image),
        cv2.COLOR_RGB2BGR,
    )

    overlay_bgr = overlay_heatmap(
        original_bgr,
        heatmap,
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return overlay_rgb


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():

    st.set_page_config(
        page_title="AI Skin Disease Detection",
        layout="wide",
    )

    st.title(
        "AI-Based Skin Disease Detection "
        "and Affected-Area Estimation"
    )

    st.caption(
        "Academic prototype. The classifier predicts one of the "
        "7 HAM10000 classes. Cancer Status is a broad class grouping, "
        "not a medical diagnosis. Affected Area/Level is an image-based "
        "project indicator, not clinical disease severity."
    )

    # ------------------------------------------------------------------------
    # IMAGE UPLOAD
    # ------------------------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a skin image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to begin."
        )
        return

    try:
        pil_image = Image.open(
            uploaded_file,
        ).convert("RGB")

    except Exception as e:
        st.error(
            f"Could not open image: {e}"
        )
        return

    # ------------------------------------------------------------------------
    # DISPLAY IMAGE
    # ------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            pil_image,
            use_container_width=True,
        )

    # ------------------------------------------------------------------------
    # LOAD MODELS
    # ------------------------------------------------------------------------

    try:

        with st.spinner(
            "Loading AI models..."
        ):

            classifier, idx_to_class = (
                load_classifier()
            )

            seg_model = (
                load_segmentation_model()
            )

    except Exception as e:

        st.error(
            "Model loading failed."
        )

        st.exception(e)

        return

    # ------------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------------

    try:

        with st.spinner(
            "Running 7-class classification..."
        ):

            (
                predicted_class,
                confidence,
                arr_preprocessed,
                preds,
            ) = run_classification(
                pil_image,
                classifier,
                idx_to_class,
            )

            pred_index = int(
                np.argmax(preds)
            )

    except Exception as e:

        st.error(
            "Classification failed."
        )

        st.exception(e)

        return

    # ------------------------------------------------------------------------
    # SEGMENTATION
    # ------------------------------------------------------------------------

    try:

        with st.spinner(
            "Running lesion segmentation..."
        ):

            (
                binary_mask,
                raw_mask,
            ) = run_segmentation(
                pil_image,
                seg_model,
            )

            affected_pct = (
                compute_affected_area_percentage(
                    binary_mask,
                )
            )

            band = severity_band(
                affected_pct,
                LOW_THRESHOLD,
                HIGH_THRESHOLD,
            )

    except Exception as e:

        st.error(
            "Segmentation failed."
        )

        st.exception(e)

        return

    # ------------------------------------------------------------------------
    # GRAD-CAM
    # ------------------------------------------------------------------------

    gradcam_overlay = None

    try:

        with st.spinner(
            "Generating Grad-CAM explanation..."
        ):

            gradcam_overlay = run_gradcam(
                arr_preprocessed,
                classifier,
                pil_image,
                pred_index,
            )

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

    # ------------------------------------------------------------------------
    # SEGMENTATION MASK
    # ------------------------------------------------------------------------

    with col2:

        st.subheader(
            "Segmentation Mask"
        )

        st.image(
            binary_mask * 255,
            use_container_width=True,
            clamp=True,
        )

    # ------------------------------------------------------------------------
    # RESULT VARIABLES
    # ------------------------------------------------------------------------

    full_name = CLASS_FULL_NAMES.get(
        predicted_class,
        predicted_class,
    )

    cancer_status = get_cancer_status(
        predicted_class,
    )

    # ------------------------------------------------------------------------
    # MAIN RESULTS
    # ------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Classification Results"
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:

        st.metric(
            "Predicted Condition",
            full_name,
        )

    with r2:

        st.metric(
            "Cancer Status",
            cancer_status,
        )

    with r3:

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%",
        )

    with r4:

        st.metric(
            "Affected Area",
            f"{affected_pct:.1f}%",
        )

    # ------------------------------------------------------------------------
    # CLEAR CANCER STATUS MESSAGE
    # ------------------------------------------------------------------------

    st.markdown(
        "### Cancer Status"
    )

    if predicted_class in {"mel", "bcc"}:

        st.error(
            f"**{cancer_status}**\n\n"
            f"Predicted class: **{predicted_class} — {full_name}**\n\n"
            "This is a model classification result and must not be "
            "treated as a confirmed cancer diagnosis."
        )

    elif predicted_class == "akiec":

        st.warning(
            f"**{cancer_status}**\n\n"
            f"Predicted class: **{predicted_class} — {full_name}**\n\n"
            "This class is associated with a premalignant/pre-cancerous "
            "lesion category. This model result is not a diagnosis."
        )

    else:

        st.success(
            f"**{cancer_status}**\n\n"
            f"Predicted class: **{predicted_class} — {full_name}**"
        )

    # ------------------------------------------------------------------------
    # HAM10000 CLASS CODE
    # ------------------------------------------------------------------------

    st.info(
        f"HAM10000 class: **{predicted_class}**"
    )

    # ------------------------------------------------------------------------
    # ALL 7 CLASS PROBABILITIES
    # ------------------------------------------------------------------------

    st.markdown(
        "### Disease Classification Probabilities"
    )

    probability_data = []

    for i, prob in enumerate(preds):

        class_code = idx_to_class[i]

        probability_data.append(
            {
                "Class": class_code,
                "Condition": CLASS_FULL_NAMES.get(
                    class_code,
                    class_code,
                ),
                "Probability": float(prob) * 100.0,
            }
        )

    probability_data.sort(
        key=lambda item: item["Probability"],
        reverse=True,
    )

    for item in probability_data:

        class_code = item["Class"]
        condition = item["Condition"]
        probability = item["Probability"]

        st.write(
            f"**{condition} ({class_code}) — "
            f"{probability:.2f}%**"
        )

        st.progress(
            min(
                max(
                    int(probability),
                    0,
                ),
                100,
            )
        )

    # ------------------------------------------------------------------------
    # AFFECTED AREA
    # ------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Affected-Area Result"
    )

    a1, a2 = st.columns(2)

    with a1:

        st.metric(
            "Affected Area",
            f"{affected_pct:.1f}%",
        )

    with a2:

        st.metric(
            "Project-Defined Affected Level",
            str(band),
        )

    st.caption(
        f"Project thresholds: Low < {LOW_THRESHOLD:.0f}%, "
        f"Moderate {LOW_THRESHOLD:.0f}–{HIGH_THRESHOLD:.0f}%, "
        f"High > {HIGH_THRESHOLD:.0f}%. "
        "These thresholds are project-defined and are not clinical severity levels."
    )

    # ------------------------------------------------------------------------
    # GRAD-CAM
    # ------------------------------------------------------------------------

    if gradcam_overlay is not None:

        st.divider()

        st.subheader(
            "Grad-CAM Explanation"
        )

        st.image(
            gradcam_overlay,
            use_container_width=True,
            caption=(
                "Heatmap showing image regions that influenced "
                "the classification prediction."
            ),
        )

    # ------------------------------------------------------------------------
    # FINAL DISCLAIMER
    # ------------------------------------------------------------------------

    st.divider()

    st.warning(
        "⚠️ Disclaimer: This is an academic and decision-support "
        "prototype. The predicted condition and cancer-status grouping "
        "are model outputs and are not a medical diagnosis. The affected "
        "area and affected level are image-based project indicators and "
        "must not be interpreted as clinical disease severity. Consult a "
        "qualified healthcare professional for diagnosis and treatment."
    )


# ============================================================================
# START
# ============================================================================

if __name__ == "__main__":
    main()
