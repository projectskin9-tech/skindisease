"""
gradcam.py
==========
Grad-CAM explainability for the trained classification model.
Generates a heatmap showing which regions of the image most influenced
the predicted disease category, and overlays it on the original image.

Can be used as a standalone script or imported into the Streamlit app.

Usage (standalone):
    python gradcam.py --model outputs/mobilenet_final.h5 --image sample.jpg \
        --last_conv_layer Conv_1 --class_indices outputs/class_indices.json
"""

import json
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def _get_output_ndim(layer):
    """Get output rank in a way that works across Keras versions."""
    try:
        # Newer Keras (3.x): layer.output is a KerasTensor with .shape
        shape = layer.output.shape
        return len(shape)
    except Exception:
        try:
            # Older Keras (2.x): layer.output_shape is a tuple
            return len(layer.output_shape)
        except Exception:
            return 0


def find_last_conv_layer(model):
    """Auto-detect the last convolutional layer name if not provided."""
    for layer in reversed(model.layers):
        # Base models are often nested; search inside if needed
        if isinstance(layer, tf.keras.Model):
            nested = find_last_conv_layer(layer)
            if nested:
                return nested
        if _get_output_ndim(layer) == 4:  # (batch, H, W, C) -> conv-like
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image, shape (1, H, W, 3)
    model: full Keras classification model
    last_conv_layer_name: name of last conv layer (inside nested base model if applicable)
    """
    # Build a model that maps input -> (last conv output, final predictions)
    # NOTE: use model.input (singular) rather than model.inputs (list) --
    # in Keras 3, building a new Functional model from model.inputs (a list)
    # can cause a structural mismatch when called with a plain array.
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        if last_conv_layer_name in [l.name for l in model.layers]
        else [_find_nested_layer_output(model, last_conv_layer_name), model.output],
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def _find_nested_layer_output(model, layer_name):
    """Search nested sub-models (e.g. a MobileNetV2 base) for a layer by name."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(layer_name).output
            except ValueError:
                continue
    raise ValueError(f"Layer {layer_name} not found in model or nested sub-models.")


def overlay_heatmap(original_img_bgr, heatmap, alpha=0.4):
    """
    original_img_bgr: original image as read by cv2 (BGR, uint8), any size
    heatmap: 2D array in [0, 1]
    Returns overlayed image (BGR, uint8) resized to original image size.
    """
    h, w = original_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_img_bgr, 1 - alpha, 0)
    return overlay


def preprocess_for_model(image_path, model_name="mobilenet", target_size=(224, 224)):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    arr = np.expand_dims(img_resized.astype(np.float32), axis=0)

    if model_name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError("Unknown model_name")

    arr = preprocess_input(arr)
    return arr, img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained .h5 classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_name", type=str, default="mobilenet",
                         choices=["mobilenet", "resnet50", "efficientnet"])
    parser.add_argument("--last_conv_layer", type=str, default=None,
                         help="Name of last conv layer; auto-detected if omitted")
    parser.add_argument("--class_indices", type=str, required=True,
                         help="Path to class_indices.json produced during training")
    parser.add_argument("--output", type=str, default="gradcam_output.jpg")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)

    with open(args.class_indices) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    last_conv_layer_name = args.last_conv_layer or find_last_conv_layer(model)
    print(f"Using last conv layer: {last_conv_layer_name}")

    img_array, original_bgr = preprocess_for_model(args.image, args.model_name)
    heatmap, pred_index, probs = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    predicted_class = idx_to_class[pred_index]
    confidence = float(probs[pred_index]) * 100
    print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")

    overlay = overlay_heatmap(original_bgr, heatmap)
    cv2.imwrite(args.output, overlay)
    print(f"Saved Grad-CAM overlay to {args.output}")


if __name__ == "__main__":
    main()