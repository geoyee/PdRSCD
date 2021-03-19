import os
import numpy as np
import paddle
from paddle.io import DataLoader
from ppcd.metrics import ComputAccuracy


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        loss_list.append(losses['coef'][i] * loss_i(logits, labels))
    return loss_list


def Train(model, 
          epoch,
          batch_size,
          train_data, 
          eval_data=None, 
          optimizer=None,
          losses=None,
          save_model_path=None,
          save_epoch=2,
          log_batch=10):
    # 数据读取器
    data_lens = len(train_data) // batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    if eval_data is not None:
        eval_loader = DataLoader(eval_data, batch_size=batch_size, drop_last=True)
    # 开始训练
    if save_model_path is not None:
        if os.path.exists(save_model_path) == False:
            os.mkdir(save_model_path)
    for epoch_id in range(epoch):
        model.train()
        for batch_id, (A_img, B_img, lab) in enumerate(train_loader()):
            pred_list = model(A_img, B_img)
            # img = paddle.concat([A_img, B_img], axis=1)
            # pred_list = model(img)
            loss_list = loss_computation(
                logits_list=pred_list,
                labels=lab,
                losses=losses)
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if (batch_id + 1) % log_batch == 0:
                print("[Train] epoch: {}, batch: {}, loss: {:.4f}".format(epoch_id + 1, batch_id + 1, loss.numpy()[0]))
            if ((epoch_id + 1) % save_epoch) == 0 and (batch_id == (data_lens - 1)):
                model.eval()
                val_losses = []
                val_mious = []
                val_accs = []
                val_kappas = []
                for val_A_img, val_B_img, val_lab in eval_loader():
                    val_pred_list = model(val_A_img, val_B_img)
                    val_lab = val_lab.astype('int32')
                    # val_img = paddle.concat([val_A_img, val_B_img], axis=1)
                    # val_pred_list = model(val_img)
                    val_loss_list = loss_computation(
                        logits_list=val_pred_list,
                        labels=val_lab,
                        losses=losses)
                    val_loss = sum(val_loss_list)
                    val_losses.append(val_loss.numpy())
                    pred = paddle.argmax(val_pred_list[0], axis=1, keepdim=True, dtype='int32')
                    val_miou, val_acc, val_kappa = ComputAccuracy(pred, val_lab)
                    val_mious.append(val_miou)
                    val_accs.append(val_acc)
                    val_kappas.append(val_kappa)
                print("[Eval] epoch: {}, loss: {:.4f}, miou: {:.4f}, acc: {:.4f}, kappa: {:.4f}" \
                      .format(epoch_id + 1, np.mean(val_losses), np.mean(val_mious), \
                      np.mean(val_accs), np.mean(val_kappas)))
                paddle.save(model.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdparams')
                paddle.save(optimizer.state_dict(), os.path.join(save_model_path, 'epoch_' + str(epoch_id)) + '.pdopt')