from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, stride, drop_rate=0.5,activate=nn.ReLU()):
        '''
        channels:列表或元组，每层的输入通道数和输出通道数
        '''
        super(VGGBlock, self).__init__()
        self.layers = nn.ModuleList()  # 实现了将添加进来的层参数进行注册
        for i in range(len(channels) - 1):  # 遍历
            self.layers.append(nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                padding=padding,
                kernel_size=kernel_size,
                stride=stride))
            self.layers.append(activate)
            self.layers.append(nn.BatchNorm2d(num_features=channels[i + 1]))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, num_classes,drop_rate=0.5,activate=nn.ReLU()):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.activate=activate
        # block1
        self.block1 = VGGBlock([3, 64, 64], padding=1, kernel_size=3, stride=1,activate=activate)
        # block2
        self.block2 = VGGBlock([64, 128, 128], padding=1, kernel_size=3, stride=1,activate=activate)
        # block3
        self.block3 = VGGBlock([128, 256, 256, 256], padding=1, kernel_size=3, stride=1,activate=activate)
        # block4
        self.block4 = VGGBlock([256, 512, 512, 512], padding=1, kernel_size=3, stride=1,activate=activate)
        # block5
        self.block5 = VGGBlock([512, 512, 512, 512], padding=1, kernel_size=3, stride=1,activate=activate)

        # classifier
        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.fc3 = nn.Linear(in_features=4096, out_features=self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.view(x.size(0), -1)

        # classifier
        x = self.drop1(self.activate(self.fc1(x)))
        x = self.drop2(self.activate(self.fc2(x)))
        x = self.fc3(x)
        return x