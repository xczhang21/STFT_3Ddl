from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import torch
from utilities.metrics import MaskedMSEMetric, MaskedMAEMetric, MaskedRMSEMetric

def compute_metrics(y_true, y_pred, num_classes):
    """
    计算 Accuracy, Precision, Recall, F1-score
    :param y_true: 真实标签 (Tensor)
    :param y_pred: 预测标签 (Tensor)
    :param num_classes: 类别数
    :return: dict {'accuracy':, 'precision':, 'recall':, 'f1_score':}
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def compute_map(y_true, y_scores, num_classes):
    """
    计算 Mean Average Precision (mAP)
    :param y_true: 真实标签 (Tensor) - One-hot 形式
    :param y_scores: 预测分数 (Tensor) - Softmax 输出
    :param num_classes: 类别数
    :return: mAP 值
    """
    y_true = y_true.cpu().numpy()
    y_scores = y_scores.detach().cpu().numpy()

    ap_list = []
    for i in range(num_classes):
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        ap_list.append(ap)

    return np.mean(ap_list)

def log_metrics(writer, prefix, loss, metrics, train_map, epoch_num):
    
    writer.add_scalar(f'1-Loss/{prefix}', loss, epoch_num)
    writer.add_scalar(f'2-Accuracy/{prefix}', metrics['accuracy'], epoch_num)
    writer.add_scalar(f'3-Precision/{prefix}', metrics['precision'], epoch_num)
    writer.add_scalar(f'4-Recall/{prefix}', metrics['recall'], epoch_num)
    writer.add_scalar(f'5-F1_score/{prefix}', metrics['f1_score'], epoch_num)
    writer.add_scalar(f'6-mAP/{prefix}', train_map, epoch_num)


def compute_masked_metrics(preds, targets, masks):
    """
    用于 Masked 重建任务，输入整个 epoch 的所有预测/标签/掩码张量，计算平均 MSE、MAE、RMSE。
    所有输入为拼接后的 [N, C, H, W] 张量
    """
    assert preds.shape == targets.shape == masks.shape, "preds, targets, and masks must have the same shape"
    
    mse_metric = MaskedMSEMetric()
    mae_metric = MaskedMAEMetric()
    rmse_metric = MaskedRMSEMetric()

    with torch.no_grad():
        mse = mse_metric(preds, targets, masks)
        mae = mae_metric(preds, targets, masks)
        rmse = rmse_metric(preds, targets, masks)

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": rmse.item()
    }

def log_masked_metrics(writer, prefix, loss, metrics_dict, epoch_num, mask_ratio=None):
    """
    记录自监督重建任务中的loss、 MAE、RMSE、MSE 以及 mask_ratio
    """
    writer.add_scalar(f'1-Loss/{prefix}', loss, epoch_num)
    writer.add_scalar(f'2-MSE/{prefix}', metrics_dict["mse"], epoch_num)
    writer.add_scalar(f'3-MAE/{prefix}', metrics_dict["mae"], epoch_num)
    writer.add_scalar(f'4-RMSE/{prefix}', metrics_dict["rmse"], epoch_num)
    if mask_ratio is not None:
        writer.add_scalar(f'5-MaskRatio/{prefix}', mask_ratio, epoch_num)
