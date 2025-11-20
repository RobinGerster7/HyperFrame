from sklearn import metrics
import numpy as np


def evaluation_metrics(
    gt_flat: np.ndarray,
    detection_map_flat: np.ndarray,
    fpr_range: tuple[float, float] = (0,1)
) -> tuple[float, float, float, float, float]:
    """
    Computes AUC-based evaluation metrics from ROC curves over a specified FPR range.

    Args:
        gt_flat (np.ndarray): Flattened binary ground truth mask [N].
        detection_map_flat (np.ndarray): Flattened detection scores [N] in [0, 1].
        fpr_range (tuple): (min_fpr, max_fpr) range for restricting AUC(Pf, Pd) computation.

    Returns:
        tuple:
            - auc_effect: AUC(Pf, Pd) within range
            - auc_detect: AUC(τ, Pd)
            - auc_false_alarm: AUC(τ, Pf)
            - auc_oa: Overall accuracy = AUC(Pf,Pd) + AUC(τ,Pd) - AUC(τ,Pf)
            - auc_snpr: Signal-to-Noise Power Ratio = AUC(τ,Pd) / AUC(τ,Pf)
    """
    if not np.all(np.isin(gt_flat, [0, 1])):
        raise ValueError("Ground truth array must only contain binary values (0 and 1).")

    if gt_flat.shape != detection_map_flat.shape:
        raise ValueError("Ground truth and detection map must have the same shape.")

    # Compute ROC
    fpr, tpr, thresholds = metrics.roc_curve(gt_flat, detection_map_flat, drop_intermediate=False)

    # Exclude the first threshold = -inf
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    # Restrict Pf-Pd curve to given FPR range
    fpr_min, fpr_max = fpr_range
    fpr_clipped = np.clip(fpr, fpr_min, fpr_max)
    tpr_clipped = np.interp(fpr_clipped, fpr, tpr)
    valid = (fpr_clipped >= fpr_min) & (fpr_clipped <= fpr_max)
    fpr_clipped = fpr_clipped[valid]
    tpr_clipped = tpr_clipped[valid]

    if len(fpr_clipped) >= 2:
        auc_effect = float(np.trapz(tpr_clipped, fpr_clipped) / (fpr_max - fpr_min))
    else:
        auc_effect = 0.0

    # AUC(τ, Pf) and AUC(τ, Pd) remain global (over thresholds)
    auc_false_alarm = float(metrics.auc(thresholds, fpr))
    auc_detect = float(metrics.auc(thresholds, tpr))
    auc_oa = auc_effect + auc_detect - auc_false_alarm
    auc_snpr = (auc_detect / auc_false_alarm) if auc_false_alarm > 0 else float("inf")

    return (
        round(auc_effect, 5),
        round(auc_detect, 5),
        round(auc_false_alarm, 5),
        round(auc_oa, 5),
        round(auc_snpr, 5)
    )
