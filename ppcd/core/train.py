import os
import numpy as np
import paddle
# from paddle.io import DataLoader
from ppcd.datasets import DataLoader
from ppcd.utils import loss_computation
from ppcd.utils import TimeAverager, calculate_eta
from visualdl import LogWriter
import time
# from tqdm import tqdm
from ppcd.core.eval import Eval


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
    # dataloader = CDataLoader if loader == 'CDataLoader' else DataLoader
    dataloader = DataLoader
    data_lens = len(train_data) // batch_size  # 训练数据数
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
    # 开始训练
    with LogWriter(logdir=("./log/" + str(time.mktime(time.localtime())))) as writer:
        iters = 0
        for epoch_id in range(epoch): 
            model.train()
            train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True)  # 数据读取器
            for batch_id, train_load_data in enumerate(train_loader):
                batch_start = time.time()  # batch计时
                if train_load_data is None:
                    break
                (A_img, B_img, lab) = train_load_data
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
                batch_cost_averager.record((time.time() - batch_start), num_samples=batch_size)
                if (batch_id + 1) % log_batch == 0:
                    avg_train_batch_cost = batch_cost_averager.get_average()
                    eta = calculate_eta((epoch * data_lens - iters), avg_train_batch_cost)
                    print("[Train] epoch: {}, batch: {}, loss: {:.4f}, ips: {:.4f}, ETA: {}".format(
                        epoch_id + 1, batch_id + 1, loss.numpy()[0], \
                        batch_cost_averager.get_ips_average(), eta))
                    writer.add_scalar(tag="train/loss", step=iters, value=loss.numpy()[0])
                    batch_cost_averager.reset()
                if ((epoch_id + 1) % save_epoch) == 0 and (batch_id == (data_lens - 1)) and \
                   eval_data is not None:
                    val_losses, val_mious, val_class_miou, val_maccs, val_class_acc, val_mf1s, val_class_f1, val_kappas = Eval(
                        model=model,
                        eval_data=eval_data,
                        losses=losses,
                        threshold=threshold,
                        show_result=False
                    )
                    print("[Eval] epoch: {}, loss: {:.4f}, miou: {:.4f}, class_miou: {}, acc: {:.4f}, class_acc: {}, f1: {:.4f}, class_f1: {}, kappa: {:.4f}" \
                          .format(epoch_id + 1, np.mean(val_losses), np.mean(val_mious), \
                          str(np.round(val_class_miou, 4)), np.mean(val_maccs), \
                          str(np.round(val_class_acc, 4)), np.mean(val_mf1s), \
                          str(np.round(val_class_f1, 4)), np.mean(val_kappas)))
                    writer.add_scalar(tag="eval/loss", step=iters, value=np.mean(val_losses))
                    writer.add_scalar(tag="eval/acc", step=iters, value=np.mean(val_maccs))
                    writer.add_scalar(tag="eval/miou", step=iters, value=np.mean(val_mious))
                    writer.add_scalar(tag="eval/f1", step=iters, value=np.mean(val_mf1s))
                    writer.add_scalar(tag="eval/kappa", step=iters, value=np.mean(val_kappas))
                    paddle.save(model.state_dict(), os.path.join(save_model_path, 'epoch_' + \
                                str(epoch_id)) + '.pdparams')
                    paddle.save(optimizer.state_dict(), os.path.join(save_model_path, 'epoch_' + \
                                str(epoch_id)) + '.pdopt')