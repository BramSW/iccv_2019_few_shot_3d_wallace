import numpy as np

def iou(true_voxels, pred_voxels, threshold=0.4):
    bool_true_voxels = true_voxels > threshold
    bool_pred_voxels = pred_voxels > threshold
    total_union = (bool_true_voxels | bool_pred_voxels).sum()
    total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
    return (total_intersection / total_union)


def iou_list(true_voxels, pred_voxels, threshold=0.4):
    bool_true_voxels = true_voxels > threshold
    bool_pred_voxels = pred_voxels > threshold
    scores = []
    for true_vox, pred_vox in zip(bool_true_voxels, bool_pred_voxels):
        union = (true_vox | pred_vox).sum()
        intersection = (true_vox & pred_vox).sum()
        score = intersection / union
        scores.append(score)
    return scores
