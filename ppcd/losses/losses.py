import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BCELoss(nn.Layer):
    def __init__(self,
                 weight=None,
                 pos_weight=None,
                 ignore_index=255,
                 edge_label=False):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.edge_label = edge_label
        self.EPS = 1e-10
        if self.weight is not None:
            if isinstance(self.weight, str):
                if self.weight != 'dynamic':
                    raise ValueError(
                        "if type of `weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.weight))
            elif isinstance(self.weight, paddle.VarBase):
                raise TypeError(
                    'The type of `weight` is wrong, it should be Tensor or str, but it is {}'
                    .format(type(self.weight)))
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, str):
                if self.pos_weight != 'dynamic':
                    raise ValueError(
                        "if type of `pos_weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.pos_weight))
            elif isinstance(self.pos_weight, float):
                self.pos_weight = paddle.to_tensor(
                    self.pos_weight, dtype='float32')
            else:
                raise TypeError(
                    'The type of `pos_weight` is wrong, it should be float or str, but it is {}'
                    .format(type(self.pos_weight)))

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, C), where each
                value is 0 or 1, and if shape is more than 2D, this is
                (N, C, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
        mask = (label != self.ignore_index)
        mask = paddle.cast(mask, 'float32')
        # label.shape should equal to the logit.shape
        if label.shape[1] != logit.shape[1]:
            label = label.squeeze(1)
            label = F.one_hot(label, logit.shape[1])
            label = label.transpose((0, 3, 1, 2))
        if isinstance(self.weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            weight_pos = 2 * neg_num / (sum_num + self.EPS)
            weight_neg = 2 * pos_num / (sum_num + self.EPS)
            weight = weight_pos * label + weight_neg * (1 - label)
        else:
            weight = self.weight
        if isinstance(self.pos_weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            pos_weight = 2 * neg_num / (sum_num + self.EPS)
        else:
            pos_weight = self.pos_weight
        label = label.astype('float32')
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            weight=weight,
            reduction='none',
            pos_weight=pos_weight)
        loss = loss * mask
        loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        label.stop_gradient = True
        mask.stop_gradient = True
        return loss


class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = paddle.unsqueeze(labels, 1)
        num_classes = logits.shape[1]
        mask = (labels != self.ignore_index)
        logits = logits * mask
        labels = paddle.cast(labels, dtype='int32')
        single_label_lists = []
        for c in range(num_classes):
            single_label = paddle.cast((labels == c), dtype='int32')
            single_label = paddle.squeeze(single_label, axis=1)
            single_label_lists.append(single_label)
        labels_one_hot = paddle.stack(tuple(single_label_lists), axis=1)
        logits = F.softmax(logits, axis=1)
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')
        dims = (0,) + tuple(range(2, labels.ndimension()))
        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss


class MixedLoss(nn.Layer):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.

    Args:
        losses (list of nn.Layer): A list consisting of multiple loss classes
        coef (float|int): Weighting coefficient of multiple loss

    Returns:
        A callable object of MixedLoss.
    """
    def __init__(self, losses, coef):
        super(MixedLoss, self).__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))
        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        final_output = 0
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels)
            final_output += output * self.coef[i]
        return final_output


class TripletLoss(nn.Layer):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, logit, label=None):
        N, fea_dim = logit.shape[:2]
        logit_norm = paddle.sqrt(paddle.sum(paddle.square(logit), axis=1)).reshape((N, 1, -1))
        logit = paddle.divide(logit, logit_norm)
        output = paddle.reshape(logit, shape=[-1, 3, fea_dim])
        anchor, positive, negative = paddle.split(output, num_or_sections=3, axis=1)
        anchor = paddle.reshape(anchor, shape=[-1, fea_dim])
        positive = paddle.reshape(positive, shape=[-1, fea_dim])
        negative = paddle.reshape(negative, shape=[-1, fea_dim])
        a_p = paddle.square(anchor - positive)
        a_n = paddle.square(anchor - negative)
        a_p = paddle.sum(a_p, axis=1)
        a_n = paddle.sum(a_n, axis=1)
        loss = F.relu(a_p + self.margin - a_n)
        return loss


class BCLoss(nn.Layer):
    """
        STANet
        batch-balanced contrastive loss
        no-change，1
        change，-1
    """
    def __init__(self, margin=2.0):
        super(BCLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = -1 * (2 * label - 1)
        # print(label, distance)
        pos_num = paddle.sum((label == 1).astype('float32')) + 0.0001
        neg_num = paddle.sum((label == -1).astype('float32')) + 0.0001
        loss_1 = paddle.sum((1 + label) / 2 * paddle.pow(distance, 2)) / pos_num
        loss_2 = paddle.sum((1 - label) / 2 * paddle.pow(paddle.clip(self.margin - distance, min=0.0), 2)) / neg_num
        loss = loss_1 + loss_2
        return loss


class ConstLoss(nn.Layer):
    # 不参与损失计算，直接返回固定损失值
    def __init__(self, value=0):
        super(ConstLoss, self).__init__()
        self.value = value

    def forward(self, img, label):
        return paddle.to_tensor(self.value)


class LabelL1Loss(nn.Layer):
    # 计算标签的l1loss
    def __init__(self):
        super(LabelL1Loss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, fc, lab):
        return self.l1loss(fc, lab)
