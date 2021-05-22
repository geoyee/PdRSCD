import os
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from ppcd.metrics import ComputAccuracy
from visualdl import LogWriter
import time


def check_logits_losses(logits_list, losses):
    if not losses.has_key('ceof'):
        losses['ceof'] = [1] * len(logits_list)
    if not losses.has_key('decay'):
        losses['decay'] = [1] * len(logits_list)
    if len(losses['type']) == len(losses['ceof']) and \
       len(losses['type']) == len(losses['decay']):
        len_logits = len(logits_list)
        len_losses = len(losses['types'])
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))
    else:
        raise RuntimeError('The logits_list type/coef/decay should equal.')


def loss_computation(logits_list, labels, losses, epoch=None, batch=None):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        coef_i = losses['coef'][i]
        loss_i = losses['types'][i]
        if isinstance(labels, list):
            label_i = labels[i]
        else:
            label_i = labels[0]
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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    if eval_data is not None:
        eval_loader = DataLoader(eval_data, batch_size=batch_size, drop_last=True)
    # 创建模型保存文件夹
    if save_model_path is not None:
        if os.path.exists(save_model_path) == False:
            os.mkdir(save_model_path)
    # 加载预训练参数
    if pre_params_path is not None:
        para_state_dict = paddle.load(pre_params_path)
        model.set_dict(para_state_dict)
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
                if (batch_id + 1) % log_batch == 0:
                    print("[Train] epoch: {}, batch: {}, loss: {:.4f}".format(epoch_id + 1, batch_id + 1, loss.numpy()[0]))
                    writer.add_scalar(tag="train/loss", step=iters, value=loss.numpy()[0])
                if ((epoch_id + 1) % save_epoch) == 0 and (batch_id == (data_lens - 1)):
                    model.eval()
                    val_losses = []
                    val_mious = []
                    val_accs = []
                    val_kappas = []
                    for val_A_img, val_B_img, val_lab in eval_loader():
                        val_pred_list = model(val_A_img, val_B_img)
                        val_lab = val_lab.astype('int64')
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
                        val_miou, val_acc, val_kappa = ComputAccuracy(val_pred, val_lab)
                        val_mious.append(val_miou)
                        val_accs.append(val_acc)
                        val_kappas.append(val_kappa)
                    print("[Eval] epoch: {}, loss: {:.4f}, miou: {:.4f}, acc: {:.4f}, kappa: {:.4f}" \
                        .format(epoch_id + 1, np.mean(val_losses), np.mean(val_mious), \
                        np.mean(val_accs), np.mean(val_kappas)))
                    writer.add_scalar(tag="eval/loss", step=iters, value=np.mean(val_losses))
                    writer.add_scalar(tag="eval/acc", step=iters, value=np.mean(val_accs))
                    writer.add_scalar(tag="eval/miou", step=iters, value=np.mean(val_mious))
                    writer.add_scalar(tag="eval/kappa", step=iters, value=np.mean(val_kappas))
                    paddle.save(model.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdparams')
                    paddle.save(optimizer.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdopt')