import torch
import numpy as np
from sipbuild.generator.outputs import output_pyi

from dataset import TrashDataset
from utils import label_to_id, process_img, show_img
import os
from model import VGGNet
import torch.nn.functional as F
from tqdm import tqdm

#Environment Variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

save_path='model/vgg16.pkl'#模型保存路径
dsize=(224,224)#图像尺寸

path='Dataset'
split_dir = os.path.join(path, "trash_split")
test_dir=os.path.join(split_dir,"test")
test_data = TrashDataset(data_dir=test_dir,class_dict=label_to_id)#构建测试数据集

# 推理时我们只需要加载模型的网络参数即可，因为不再训练，不需要优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#获取可用的设备
model=VGGNet(num_classes=len(label_to_id)).to(device)#将model转移到可用设备上
checkpoint = torch.load(save_path)#加载模型
model.load_state_dict(checkpoint['model_state_dict'])#模型参数加载

def infer(img):
    """
    负责对一张img进行推理预测
    Args:
        img:

    Returns:预测类别下标，类别概率

    """
    model.eval()
    x = process_img(img, dsize) #图像预处理
    x = torch.Tensor([x]).to(device)
    with torch.no_grad(): #不求导
        output = model(x) #获取输出值
        y_hat = F.softmax(output)[0].cpu().numpy() #softmax计算概率
    y_pred = np.argmax(y_hat) #提取预测下标
    y_p = y_hat[y_pred] #提取预测类别概率
    return y_pred, y_p

result = []
for item in tqdm(test_data): #遍历测试集
    img, y_true = item #提取img和标签
    res = infer(img)
    result.append((img, y_true, res[0], res[1])) #保存结果

show_img(result[:16])#展示结果