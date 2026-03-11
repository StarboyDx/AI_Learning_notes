import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



n_epochs = 3
batch_size_train = 64
batch_size_test=256
learning_rate = 0.01
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)


#############################定义数据集###############################
train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('Dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('Dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=batch_size_test, shuffle=True)

#############################定义网络模型###############################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#############################定义损失函数和优化器###############################
model = Net()
loss_func=torch.nn.NLLLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)

#############################搭建训练流程###############################
n_epochs = 10
# 定义计算准确率的函数
def calc_acc(output, target):
    pred = output.argmax(dim=1)
    num_correct = torch.eq(pred, target).sum().float().item()  # 计算准确率
    return num_correct, num_correct / pred.shape[0]


for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    train_num_correct = 0

    # 训练
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        num_correct, acc = calc_acc(output, target)
        train_num_correct += num_correct

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), acc))

    print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}\tLr: {:.6f}".format(epoch,
                                                                train_loss / len(train_loader),
                                                                train_num_correct / len(train_loader.dataset),
                                                                scheduler.get_lr()[0]))

    # 验证
    val_loss = 0
    val_num_correct = 0
    model.eval()  # 测试模式
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        loss = loss_func(output, target)
        val_loss += float(loss.item())
        num_correct, _ = calc_acc(output, target)
        val_num_correct += num_correct
    print("Test Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,
                                                              val_loss / len(test_loader),
                                                              val_num_correct / len(test_loader.dataset)))


