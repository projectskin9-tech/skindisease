"""
area_severity.py
=================
Computes the affected-area percentage from a segmentation mask, and maps it
to a project-defined severity band (Low / Moderate / High).

This is intentionally kept as a small, standalone module so it can be
imported by both the Streamlit app and any evaluation scripts.
"""

import numpy as np


def compute_affected_area_percentage(mask: np.ndarray, reference_mask: np.ndarray = None) -> float:
    """
    mask: binary array (H, W) where 1 = affected pixel, 0 = background.
    reference_mask: optional binary array (H, W) marking the relevant
        visible-skin region (S). If None, the entire image is treated as S,
        as described in the project report (Section 5.4).

    Returns affected-area percentage: (A / S) * 100
    """
    mask = (mask > 0.5).astype(np.uint8)
    A = np.sum(mask)

    if reference_mask is not None:
        reference_mask = (reference_mask > 0.5).astype(np.uint8)
        S = np.sum(reference_mask)
    else:
        S = mask.size

    if S == 0:
        return 0.0

    percentage = (A / S) * 100
    return float(percentage)


def severity_band(percentage: float, low_threshold: float = 10.0, high_threshold: float = 30.0) -> str:
    """
    Maps affected-area percentage into a project-defined severity band.

    Defaults match the example in the report (Section 5.5):
        Low      : < 10%
        Moderate : 10% - 30%
        High     : > 30%

    IMPORTANT: These thresholds are for demonstration only. You and your
    guide should choose and justify the final thresholds based on your
    dataset and project objectives.
    """
    if percentage < low_threshold:
        return "Low"
    elif percentage <= high_threshold:
        return "Moderate"
    else:
        return "High"


def evaluate_against_ground_truth(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> dict:
    """
    Compares the predicted affected-area percentage against a ground-truth mask,
    for cases where reference masks are available (report Section 8.3).

    Returns a dict with predicted %, ground-truth %, and absolute percentage error.
    """
    pred_pct = compute_affected_area_percentage(predicted_mask)
    gt_pct = compute_affected_area_percentage(ground_truth_mask)
    abs_error = abs(pred_pct - gt_pct)

    return {
        "predicted_percentage": pred_pct,
        "ground_truth_percentage": gt_pct,
        "absolute_percentage_error": abs_error,
    }


if __name__ == "__main__":
    # Simple self-test
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[20:40, 20:40] = 1  # 20x20 = 400 affected pixels out of 10000

    pct = compute_affected_area_percentage(dummy_mask)
    band = severity_band(pct)
    print(f"Affected area: {pct:.2f}% -> Severity band: {band}")
