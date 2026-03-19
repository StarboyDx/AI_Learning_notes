import argparse
from torch import optim
from torch import nn
import model
from configs import BasicConfigs
from process_data import prepare_vocab
import time
import torch
import numpy as  np
bc = BasicConfigs()#加载配置参数

# 获取参数
parser = argparse.ArgumentParser()#实例化parser对象
#是否验证模型
parser.add_argument('--compute-val', default=True, action='store_true', help='compute validation accuracy or not, default:None')
#训练轮次
parser.add_argument('--epoches', default=10, type=int, help='num of epoches for trainning loop, default:20')
#使用模型
parser.add_argument('--model-name', default='birnn', help='choose one model name for trainng')
args = parser.parse_args()#模型训练

def train(net, optimizer, loss_func, train_iter, val_iter, compute_val,device, epoches, load_model_dir, save_model_dir):
    print(f'>>>We are gonna tranning {net.__class__.__name__} with epoches of {epoches}<<<')
    net = net.to(device)#将模型网络放入device
    if load_model_dir:#如果给定加载模型路径，则加载模型
        net.load_state_dict(torch.load(load_model_dir))
    for epoch in range(epoches):#遍历epochs
        print(f'=>we are training epoch[{epoch+1}]...<=')
        train_l_sum, train_acc_sum, n, start, batch_count = 0.0, 0.0, 0, time.time(), 0
        for iter_num, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device) #获取一批数据
            score = net(X) #模型计算输出值
            loss = loss_func(score, y) #计算损失
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播求导
            optimizer.step() #优化参数

            train_l_sum += loss.cpu().item() #得到损失的值
            train_acc_sum += (score.argmax(dim=1) == y).sum().cpu().item() #计算准确率
            n += y.shape[0]
            batch_count += 1
            train_acc = train_acc_sum / n

            if (iter_num + 1) % 10 == 0:
                print("Train accuracy now is %.1f%%" % (round(train_acc, 3) * 100))

        if compute_val:
            net.eval()
            val_acc = []
            with torch.no_grad():
                for iter_num, (val_X, val_y) in enumerate(val_iter):
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    val_score = net(val_X)
                    val_acc.append((val_score.argmax(dim=1) == val_y).sum().cpu().item() / len(val_y))
            print("Val accuracy is %.1f%%" % (round(np.mean(val_acc), 3) * 100))
            net.train()

        print('*' * 25)
        if (epoch + 1) % 5 == 0 and save_model_dir:
            print(f'saving model into => {save_model_dir}')
            torch.save(net.state_dict(), save_model_dir)

        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc, time.time() - start))


if __name__ == '__main__':
    vocab, label_map, train_iter, val_iter, embedding_matrix = prepare_vocab(is_train=True)
    net = getattr(model, args.model_name)(
        embedding_matrix=embedding_matrix,
        num_hiddens = bc.num_hiddens, # 从外部的 bc 中读取并传入
        num_layers = bc.num_layers, # 同上
    )
    device = bc.device
    optimizer = optim.Adam(net.parameters(), lr=bc.lr, weight_decay=bc.alpha)
    loss_func = nn.CrossEntropyLoss()

    train(net=net, optimizer=optimizer, loss_func=loss_func,
          train_iter=train_iter, val_iter=val_iter,
          compute_val=args.compute_val, device=device, epoches=args.epoches,
          load_model_dir=None, save_model_dir=bc.save_model_dir[args.model_name])
