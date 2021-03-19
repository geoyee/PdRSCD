import os
import cv2
import paddle
from paddle.io import DataLoader


def Infer(model, 
          infer_data, 
          params_path=None,
          save_img_path=None):
    # 数据读取器
    infer_loader = DataLoader(infer_data, batch_size=1)
    # 开始预测
    if save_img_path is not None:
        if os.path.exists(save_img_path) == False:
            os.mkdir(save_img_path)
    model.eval()
    para_state_dict = paddle.load(params_path)
    model.set_dict(para_state_dict)
    for idx, (A_img, B_img) in enumerate(infer_loader()):
        pred_list = model(A_img, B_img)
        # img = paddle.concat([A_img, B_img], axis=1)
        # pred_list = model(img)
        for idx2, pred in enumerate(pred_list):
            save_img = (paddle.argmax(pred, axis=1).squeeze().numpy() * 255).astype('uint8')
            save_path = os.path.join(save_img_path, str(idx) + '_' + str(idx2) + '.jpg')
            print(save_path)
            cv2.imwrite(save_path, save_img)