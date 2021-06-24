import os
import numpy as np
import paddle
from ppcd.datasets import DataLoader
from ppcd.metrics import ComputAccuracy
from ppcd.utils import loss_computation
from tqdm import tqdm


def Eval(model,    
         eval_data,
         batch_size=1,
         losses=None,
         threshold=0.5,
         ignore_index=255,
         show_result=True):
    dataloader = DataLoader
    # data_lens = len(eval_data)
    model.eval()
    val_losses = []
    val_mious = []
    val_maccs = []
    val_mf1s = []
    val_kappas = []
    val_class_mious = np.array([0, 0]).astype('float64')
    val_class_accs = np.array([0, 0]).astype('float64')
    val_class_f1s = np.array([0, 0]).astype('float64')
    # if isinstance(dataloader, CDataLoader):  # 这个地方待改进
    #     eval_loader = dataloader(eval_data, batch_size=batch_size, is_val=True)
    # else:
    #     eval_loader = dataloader(eval_data, batch_size=batch_size)
    eval_loader = dataloader(eval_data, batch_size=batch_size, is_val=True)
    val_i = 0
    for val_load_data in tqdm(eval_loader):
        if val_load_data is None:
            break
        (val_A_img, val_B_img, val_lab) = val_load_data
        val_pred_list = model(val_A_img, val_B_img)
        tmp_lab = [v_lab.astype('int64') for v_lab in val_lab]
        val_lab = tmp_lab
        # val_img = paddle.concat([val_A_img, val_B_img], axis=1)
        # val_pred_list = model(val_img)
        val_loss_list = loss_computation(
            logits_list=val_pred_list,
            labels=val_lab,
            losses=losses)
        val_loss = sum(val_loss_list)
        val_losses.append(val_loss.numpy())
        num_class = val_pred_list[0].shape[1]  # eval_loader.classes_num
        if num_class != 1:
            val_pred = paddle.argmax(val_pred_list[0], axis=1, keepdim=True, \
                                        dtype='int32')
        else:
            val_pred = (val_pred_list[0] > threshold).astype('int32')
        val_miou, val_class_miou, val_macc, val_class_acc, val_mf1, val_class_f1, val_kappa = ComputAccuracy(
            val_pred, val_lab[0], num_classes=num_class, ignore_index=ignore_index)
        val_mious.append(val_miou)
        val_maccs.append(val_macc)
        val_mf1s.append(val_mf1)
        val_kappas.append(val_kappa)
        val_class_mious += val_class_miou
        val_class_accs += val_class_acc
        val_class_f1s += val_class_f1
        val_i += 1
        # print(val_class_mious, val_class_accs, val_class_f1s)
    vcm = val_class_mious / val_i
    vca = val_class_accs / val_i
    vcf = val_class_f1s / val_i
    if show_result:
        print("[Eval] loss: {:.4f}, miou: {:.4f}, class_miou: {}, acc: {:.4f}, class_acc: {}, f1: {:.4f}, class_f1: {}, kappa: {:.4f}" \
                .format(np.mean(val_losses), np.mean(val_mious), \
                str(vcm), np.mean(val_maccs), \
                str(vca), np.mean(val_mf1s), \
                str(vcf), np.mean(val_kappas)))
        return
    else:
        return val_losses, val_mious, vcm, val_maccs, \
               vca, val_mf1s, vcf, val_kappas