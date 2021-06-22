# import paddle


def check_logits_losses(logits_list, losses):
    # 自动权重和衰减
    if 'ceof' not in losses.keys():
        losses['ceof'] = [1] * len(losses['type'])
    if 'decay' not in losses.keys():
        losses['decay'] = [1] * len(losses['type'])
    if len(losses['type']) == len(losses['ceof']) and \
       len(losses['type']) == len(losses['decay']):
        len_logits = len(logits_list)
        len_losses = len(losses['type'])
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))
    else:
        raise RuntimeError('The logits_list type/coef/decay should equal.')


def loss_computation(logits_list, labels, losses, epoch=None, batch=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    lab_m = False
    if len(labels) > 1:
        lab_m = True
        if len(labels) != len(logits_list):
            raise RuntimeError(
                'The length of logits_list should equal to labels: {} != {}.'
                .format(len(logits_list), len(labels)))
    for i in range(len(logits_list)):
        logits = logits_list[i]
        coef_i = losses['ceof'][i]
        loss_i = losses['type'][i]
        label_i = labels[i] if lab_m else labels[0]  # 多标签损失
        if epoch != None and (epoch != 0 and batch == 0):
            decay_i = losses['decay'][i] ** epoch
            # print(decay_i)
            loss_list.append(decay_i * coef_i * loss_i(logits, label_i))
        else:
            loss_list.append(coef_i * loss_i(logits, label_i))
    return loss_list