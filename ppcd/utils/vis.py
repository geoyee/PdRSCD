from PIL import Image
import matplotlib.pyplot as plt


def show_result_RGB(A_path, B_path, inf_path, lab_path=None):
    # 打开图像
    A_img = Image.open(A_path)
    B_img = Image.open(B_path)
    inf_img = Image.open(inf_path)
    if lab_path is not None:
        lab_img = Image.open(lab_path)
        # 显示
        plt.figure(figsize=(10, 10))
        plt.subplot(221);plt.imshow(A_img);plt.title('time1')
        plt.subplot(222);plt.imshow(B_img);plt.title('time2')
        plt.subplot(223);plt.imshow(inf_img);plt.title('infer')
        plt.subplot(224);plt.imshow(lab_img);plt.title('label')
    else:
        plt.figure(figsize=(15, 5))
        plt.subplot(131);plt.imshow(A_img);plt.title('time1')
        plt.subplot(132);plt.imshow(B_img);plt.title('time2')
        plt.subplot(133);plt.imshow(inf_img);plt.title('infer')
    plt.show()