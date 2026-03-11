import os
from shutil import copyfile #拷贝文件
import random #实现随机分割
from tqdm import tqdm #进度条展示处理进度

path = 'Dataset' #数据一级目录
random.seed(1) #设定随机数种子

dataset_dir = os.path.join(path, "trash") #源数据路径
split_dir = os.path.join(path, "trash_split") #分割数据目录
split_dirs = [os.path.join(split_dir, name) for name in ["train", "valid", "test"]] #三个子集的目录
print(os.listdir(dataset_dir)) #打印目录观察
# 设定分割比例
train_pct = 0.8
valid_pct = 0.15

# 遍历目录
for root, dirs, files in os.walk(dataset_dir):
    for sub_dir in dirs:  # 提取子目录(类别)
        imgs = os.listdir(os.path.join(root, sub_dir))
        imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))  # 获取所有图片文件名称
        random.shuffle(imgs)  # 打乱数据
        img_count = len(imgs)  # 总数居数

        train_point = int(img_count * train_pct)  # 计算分割点下标
        valid_point = int(img_count * (train_pct + valid_pct))
        # 分割imgs
        images = [imgs[:train_point], imgs[train_point:valid_point], imgs[valid_point:]]
        out_dirs = [os.path.join(dir_, sub_dir) for dir_ in split_dirs]  # 拼接目录
        for image, out_dir in zip(images, out_dirs):
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)  # 创建目录
            for img in tqdm(image, desc=out_dir):
                target_path = os.path.join(out_dir, img) #目标路径
                src_path = os.path.join(dataset_dir, sub_dir, img)#源路径
                copyfile(src_path, target_path)  # 拷贝图片

        print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir,
                                                             train_point,
                                                             valid_point - train_point,
                                                             img_count - valid_point))