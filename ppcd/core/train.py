import os
import numpy as np
import paddle
from ppcd.datasets import CDataLoader
from ppcd.metrics import ComputAccuracy
from ppcd.utils import TimeAverager, calculate_eta
from visualdl import LogWriter
import time


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


def Train(model, 
          epoch,
          batch_size,
          train_data, 
          eval_data=None, 
          optimizer=None,
          losses=None,
          pre_params_path=None,
          save_model_path=None,
          save_epoch=2,
          log_batch=10,
          threshold=0.5):
    # 数据读取器
    data_lens = len(train_data) // batch_size
    train_loader = CDataLoader(train_data, batch_size=batch_size)
    if eval_data is not None:
        eval_loader = CDataLoader(eval_data, batch_size=batch_size)
    # 创建模型保存文件夹
    if save_model_path is not None:
        if os.path.exists(save_model_path) == False:
            os.mkdir(save_model_path)
    # 加载预训练参数
    if pre_params_path is not None:
        para_state_dict = paddle.load(pre_params_path)
        model.set_dict(para_state_dict)
    # 计时
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    # 开始训练
    with LogWriter(logdir=("./log/" + str(time.mktime(time.localtime())))) as writer:
        iters = 0
        for epoch_id in range(epoch):
            model.train()
            for batch_id, (A_img, B_img, lab) in enumerate(train_loader()):
                iters += 1
                pred_list = model(A_img, B_img)
                # img = paddle.concat([A_img, B_img], axis=1)
                # pred_list = model(img)
                loss_list = loss_computation(
                    logits_list=pred_list,
                    labels=lab,
                    losses=losses,
                    epoch=epoch_id,
                    batch=batch_id)
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                batch_cost_averager.record(time.time() - batch_start, num_samples=batch_size)
                if (batch_id + 1) % log_batch == 0:
                    avg_train_batch_cost = batch_cost_averager.get_average()
                    eta = calculate_eta((epoch * data_lens - iters), avg_train_batch_cost)
                    print("[Train] epoch: {}, batch: {}, loss: {:.4f}, ips: {:.4f}, ETA: {}".format(
                        epoch_id + 1, batch_id + 1, loss.numpy()[0], batch_cost_averager.get_ips_average(), eta))
                    writer.add_scalar(tag="train/loss", step=iters, value=loss.numpy()[0])
                    batch_cost_averager.reset()
                if ((epoch_id + 1) % save_epoch) == 0 and (batch_id == (data_lens - 1)):
                    model.eval()
                    val_losses = []
                    val_mious = []
                    val_accs = []
                    val_kappas = []
                    for val_A_img, val_B_img, val_lab in eval_loader():
                        val_pred_list = model(val_A_img, val_B_img)
                        val_lab = val_lab[0].astype('int64')
                        # val_img = paddle.concat([val_A_img, val_B_img], axis=1)
                        # val_pred_list = model(val_img)
                        val_loss_list = loss_computation(
                            logits_list=val_pred_list,
                            labels=val_lab,
                            losses=losses)
                        val_loss = sum(val_loss_list)
                        val_losses.append(val_loss.numpy())
                        num_class = pred_list[0].shape[1]
                        if num_class != 1:
                            val_pred = paddle.argmax(val_pred_list[0], axis=1, keepdim=True, dtype='int64')
                        else:
                            val_pred = (val_pred_list[0] > threshold).astype('int64')
                        val_miou, val_class_miou, val_acc, val_class_acc, val_kappa = ComputAccuracy(val_pred, val_lab)
                        val_mious.append(val_miou)
                        val_accs.append(val_acc)
                        val_kappas.append(val_kappa)
                    print("[Eval] epoch: {}, loss: {:.4f}, miou: {:.4f}, class_miou: {}, \
                          acc: {:.4f}, class_acc: {}, kappa: {:.4f}" \
                          .format(epoch_id + 1, np.mean(val_losses), np.mean(val_mious), \
                          str(np.round(val_class_miou, 4)), np.mean(val_accs), \
                          str(np.round(val_class_acc, 4)), np.mean(val_kappas)))
                    writer.add_scalar(tag="eval/loss", step=iters, value=np.mean(val_losses))
                    writer.add_scalar(tag="eval/acc", step=iters, value=np.mean(val_accs))
                    writer.add_scalar(tag="eval/miou", step=iters, value=np.mean(val_mious))
                    writer.add_scalar(tag="eval/kappa", step=iters, value=np.mean(val_kappas))
                    paddle.save(model.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdparams')
                    paddle.save(optimizer.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdopt')