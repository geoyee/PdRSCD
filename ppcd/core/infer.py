import os
import cv2
import paddle
from ppcd.datasets import CDataLoader


def Infer(model, 
          infer_data, 
          params_path=None,
          save_img_path=None,
          threshold=0.5):
    # 数据读取器
    infer_loader = CDataLoader(infer_data, batch_size=1)
    # 开始预测
    if save_img_path is not None:
        if os.path.exists(save_img_path) == False:
            os.mkdir(save_img_path)
    model.eval()
    para_state_dict = paddle.load(params_path)
    model.set_dict(para_state_dict)
    for A_img, B_img, name in infer_loader():
        pred_list = model(A_img, B_img)
        # img = paddle.concat([A_img, B_img], axis=1)
        # pred_list = model(img)
        num_class, H, W = pred_list[0].shape[1:]
        if num_class != 1:
            save_img = (paddle.argmax(pred_list[0], axis=1).squeeze().numpy() * 255).astype('uint8')
        else:
            save_img = ((pred_list[0] > threshold).numpy().astype('uint8') * 255).reshape([H, W])
        save_path = os.path.join(save_img_path, (name + '.jpg'))
        print(save_path)
        cv2.imwrite(save_path, save_img)