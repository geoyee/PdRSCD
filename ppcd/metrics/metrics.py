import numpy as np
import paddle
import paddle.nn.functional as F


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = paddle.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = paddle.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape, label.shape))
    # print(set(pred.numpy().flatten()))
    # print(set(label.numpy().flatten()))
    # Delete ignore_index
    mask = label != ignore_index
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = F.one_hot(pred, num_classes + 1)
    label = F.one_hot(label, num_classes + 1)
    pred = pred[:, :, :, 1:]
    label = label[:, :, :, 1:]
    pred_area = []
    label_area = []
    intersect_area = []
    for i in range(num_classes):
        pred_i = pred[:, :, :, i]
        label_i = label[:, :, :, i]
        pred_area_i = paddle.sum(pred_i)
        label_area_i = paddle.sum(label_i)
        intersect_area_i = paddle.sum(pred_i * label_i)
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)
    pred_area = paddle.concat(pred_area)
    label_area = paddle.concat(label_area)
    intersect_area = paddle.concat(intersect_area)
    return intersect_area, pred_area, label_area


def get_mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def get_accuracy_f1(intersect_area, pred_area, label_area):
    """
    Calculate accuracy
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The GT area on all classes.
    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
        np.ndarray: recall on all classes.
        float: mean recall.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    class_acc = []
    class_rcl = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        if label_area[i] == 0:
            recall = 0
        else:
            recall = intersect_area[i] / label_area[i]
        class_acc.append(acc)
        class_rcl.append(recall)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    class_acc = np.array(class_acc)
    class_rcl = np.array(class_rcl)
    f1_cls = (2 * class_acc * class_rcl) / (class_acc + class_rcl + 1e-12)
    mf1 = np.mean(f1_cls)
    return class_acc, macc, f1_cls, mf1


def get_kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return kappa


def ComputAccuracy(preds, labs, num_classes=2, ignore_index=255):
    preds = preds.astype('int32')
    labs = labs.astype('int32')
    intersect_area, pred_area, label_area = calculate_area(preds, labs, num_classes, ignore_index)
    class_iou, miou = get_mean_iou(intersect_area, pred_area, label_area)
    class_acc, macc, class_f1, mf1 = get_accuracy_f1(intersect_area, pred_area, label_area)
    kappa = get_kappa(intersect_area, pred_area, label_area)
    return miou, class_iou, macc, class_acc, mf1, class_f1, kappa