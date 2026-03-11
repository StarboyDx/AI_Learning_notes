from torch.utils.data import DataLoader #数据加载器
import torchvision.transforms as transforms #数据增强
import torch
from dataset import TrashDataset #垃圾图像数据集
from utils import label_to_id, set_seed, calc_acc #工具
import os
from model import VGGNet #网络模型
from torch import nn
from tqdm import tqdm #进度条

path='Dataset'#数据路径
n_epochs=20#训练轮次
batch_size_train = 8#训练批次
batch_size_test=16#验证批次
learning_rate = 0.0001#学习率
log_interval = 10#监控间隔
save_path='model/vgg16.pkl'#模型保存路径

set_seed()  # 设置随机种子

split_dir = os.path.join(path, "trash_split")#分割数据集路径
train_dir=os.path.join(split_dir,"train")#训练集路径
valid_dir=os.path.join(split_dir,"valid")#验证机路径

#定义预处理方法
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),#resize到固定大小
    transforms.RandomCrop(224, padding=8),#随机裁剪
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.RandomRotation(30),#随机旋转
    transforms.ToTensor(),#(H,W,C)->(C,H,W)
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#标准化
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 构建MyDataset实例
train_data = TrashDataset(data_dir=train_dir,
                          class_dict=label_to_id,
                          transform=train_transform)
valid_data = TrashDataset(data_dir=valid_dir,
                          class_dict=label_to_id,
                          transform=valid_transform)

# 构建DataLoder  num_workers=8在win下会报错  pin_memory=True表示限制只能用物理内存
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size_train,
                          shuffle=True,num_workers=0,pin_memory=True)
valid_loader = DataLoader(dataset=valid_data,
                          batch_size=batch_size_test,
                          num_workers=0,pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#获取可用的设备

model=VGGNet(num_classes=len(label_to_id)).to(device)#将model转移到可用设备上
loss_func=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)#定义优化器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)#加入了学习率调整

#判断模型文件是否存在，存在则加载模型
if os.path.exists(save_path):
    checkpoint = torch.load(save_path)#读取保存的数据
    model.load_state_dict(checkpoint['model_state_dict'])#加载模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器状态
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])#加载学习率衰减器状态
    start_epoch = checkpoint['epoch']+1#加载epoch轮次

def train_epoch():
    model.train()#设置模型为训练模式
    train_loss = 0
    train_num_correct = 0
    # 训练
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)  # x转移到可用设备
        target = target.to(device).to(torch.long)  # y转移到可用设备
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = loss_func(output, target)  # 损失计算
        loss.backward()  # 反向传播求导
        optimizer.step()  # 参数更新
        train_loss += float(loss.cpu())
        num_correct, acc = calc_acc(output, target)  # 计算准确率
        train_num_correct += num_correct

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), acc))

    print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,
                                                                train_loss / len(train_loader),
                                                                train_num_correct / len(train_loader.dataset)))

def val_epoch():
    # 验证
    val_loss = 0
    val_num_correct = 0
    model.eval()  # 设置模型为推理模式
    for data, target in tqdm(valid_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():  # 不求导
            output = model(data)
            loss = loss_func(output, target)
        val_loss += float(loss.item())
        num_correct, _ = calc_acc(output, target)
        val_num_correct += num_correct
    val_acc = val_num_correct / len(valid_loader.dataset)
    print("Val Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,
                                                              val_loss / len(valid_loader),
                                                              val_num_correct / len(valid_loader.dataset)))
    return val_acc

max_acc = 0.0 #保存模型应该选择acc最大的那一次进行保存
for epoch in range(n_epochs):
    train_epoch() #训练一个轮次
    scheduler.step() #更新学习率
    torch.cuda.empty_cache() #更新学习率

    val_acc = val_epoch()  #验证并返回验证集评估结果
    if val_acc > max_acc:
        max_acc = val_acc
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, save_path) #保存模型

