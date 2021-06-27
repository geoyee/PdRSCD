import os
import cv2
import paddle
from tqdm import tqdm
# from paddle.io import DataLoader
from ppcd.datasets import DataLoader
from ppcd.tools import splicing_list, save_tif


def Infer(model, 
          infer_data, 
          params_path=None,
          save_img_path=None,
          threshold=0.5):
    # 数据读取器
    infer_loader = DataLoader(infer_data, batch_size=1)
    # 开始预测
    if save_img_path is not None:
        if os.path.exists(save_img_path) == False:
            os.mkdir(save_img_path)
    model.eval()
    para_state_dict = paddle.load(params_path)
    model.set_dict(para_state_dict)
    lens = len(infer_data)
    for idx, infer_load_data in enumerate(infer_loader):
        if infer_load_data is None:
            break
        (A_img, B_img, name) = infer_load_data
        pred_list = model(A_img, B_img)
        # img = paddle.concat([A_img, B_img], axis=1)
        # pred_list = model(img)
        num_class, H, W = pred_list[0].shape[1:]
        if num_class == 2:
            save_img = (paddle.argmax(pred_list[0], axis=1). \
                            squeeze().numpy() * 255).astype('uint8')
        elif num_class == 1:
            save_img = ((pred_list[0] > threshold).numpy(). \
                             astype('uint8') * 255).reshape([H, W])
        else:
            save_img = (paddle.argmax(pred_list[0], axis=1). \
                            squeeze().numpy()).astype('uint8')
        save_path = os.path.join(save_img_path, (name[0] + '.jpg'))
        print('[Infer] ' + str(idx + 1) + '/' + str(lens) + ' file_path: ' + save_path)
        cv2.imwrite(save_path, save_img)


# 进行滑框预测
def Slide_Infer(model, 
                infer_data, 
                params_path=None,
                save_img_path=None,
                threshold=0.5,
                name='result'):
    # 信息修改与读取
    infer_data.out_mode = 'slide'  # 滑框模式
    raw_size = infer_data.raw_size  # 原图大小
    is_tif = infer_data.is_tif
    if infer_data.is_tif == True:
        geoinfo = infer_data.geoinfo
    # 数据读取器
    infer_loader = paddle.io.DataLoader(infer_data, batch_size=1)  # TODO:如何统一
    # 开始预测
    if save_img_path is not None:
        if os.path.exists(save_img_path) == False:
            os.mkdir(save_img_path)
    model.eval()
    para_state_dict = paddle.load(params_path)
    model.set_dict(para_state_dict)
    lens = len(infer_data)
    inf_imgs = []  # 保存块
    # for idx, infer_load_data in qenumerate(infer_loader):
    for infer_load_data in tqdm(infer_loader):
        if infer_load_data is None:
            break
        (A_img, B_img) = infer_load_data
        pred_list = model(A_img, B_img)
        # img = paddle.concat([A_img, B_img], axis=1)
        # pred_list = model(img)
        num_class, H, W = pred_list[0].shape[1:]
        if num_class == 2:
            inf_imgs.append((paddle.argmax(pred_list[0], axis=1). \
                            squeeze().numpy() * 255).astype('uint8'))
        elif num_class == 1:
            inf_imgs.append(((pred_list[0] > threshold).numpy(). \
                             astype('uint8') * 255).reshape([H, W]))
        else:
            inf_imgs.append((paddle.argmax(pred_list[0], axis=1). \
                            squeeze().numpy()).astype('uint8'))
        # print('[Infer] ' + str(idx + 1) + '/' + str(lens))
    fix_img = splicing_list(inf_imgs, raw_size)  # 拼接
    if is_tif == True:
        save_path = os.path.join(save_img_path, (name + '.tif'))
        save_tif(fix_img, geoinfo, save_path)
    else:
        save_path = os.path.join(save_img_path, (name + '.png'))
        cv2.imwrite(save_path, fix_img)