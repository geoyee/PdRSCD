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