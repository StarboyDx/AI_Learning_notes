from torch.utils.data import Dataset
import random
from PIL import Image
import os
import numpy as np
random.seed(1)

class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        猫狗分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"cat": 0, "dog": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return np.array(img), label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self,data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别子目录
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                label=self.label_name[sub_dir]
                data_info.extend([(os.path.join(root, sub_dir, img),label) for img in img_names])#构造数据(path,label)
        return data_info