import torch
import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# 类别名称映射到类别id
label_to_id = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
}

# 类别id到类别名称映射
id_to_label = {
    0:"cardboard",
    1:"glass",
    2:"metal",
    3:"paper",
    4:"plastic",
    5:"trash",
}

def set_seed(seed=1):
    """
    设置随机数种子
    Args:
        seed:

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calc_acc(output, target):
    """
    计算一批次数据的准确率accuracy
    Args:
        output: 模型预测值 (batch_size,num_classes)
        target: 真实值 (batch_size,1)

    Returns:预测正确样本个数，准确率

    """
    pred = output.argmax(dim=1)
    num_correct = torch.eq(pred, target).sum().float().item()  # 计算准确率
    return num_correct, num_correct / pred.shape[0]

def process_img(img, dsize, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
        负责在infer的对图像数据进行预处理（裁剪，归一化，转置，标准化）
        Args:
            img: 一个ndarray 要处理的图像 shape:(h,w,c)
            dsize: 裁剪的目标大小 元组 (hsize,wsize)
            mean: 三个通道的均值
            std: 三个通带的标准差

        Returns:处理后的图像数据 shape:(c,h,w)

    """
    img = cv2.resize(img, dsize = dsize) #resize到固定大小
    img = np.transpose(img, (2, 0, 1)) #维度转换 (H,W,C)->(C,H,W)
    img = img / 255 #归一化
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    img = (img - mean) / std
    return img

def show_img(imgs):
    """
    展示infer结果
    Args:
        imgs: 需要显示的图像

    Returns:

    """
    img_num = len(imgs) #获取图像个数
    rows = math.ceil(math.sqrt(img_num)) #计算行数
    cols = math.ceil(img_num / rows) #计算列数
    fig, axes = plt.subplots(nrows = rows, ncols = cols) #获取坐标系数组
    for i in range(rows):
        for j in range(cols):
            if i*cols + j >= img_num: #判断下标是否超过图像个数
                break
            img = imgs[i*cols + j] #判断下标是否超过图像个数
            axes[i][j].imshow(img[0])
            axes[i][j].set_title("y_true:%s,y_pred:%s,p:%.2f"%(id_to_label[img[1]],
                                                               id_to_label[img[2]],
                                                               img[3]))#展示输出值

    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    plt.show()