import random
import numpy as np
import time
import argparse
import os
from data_utils import DataProcessor, basic_dict
#from transformers import AdamW,get_linear_schedule_with_warmup#学习率预热 # 老版本，现在pytorch直接封装了AdamW
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from model import Encoder,AttnDecoder,Seq2Seq
from train_eval import train,evaluate,epoch_time,test,translate

parse = argparse.ArgumentParser()
parse.add_argument("--data_dir", default = 'data/', type = str, required = False,
                   help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")
parse.add_argument("--batch_size", default = 32, type = int)
parse.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
parse.add_argument("--do_test", default=False, action="store_true", help="Whether to run test.")
parse.add_argument("--do_translate", default=True, action="store_true", help="Whether to run translating.")
parse.add_argument("--learning_rate", default=5e-4, type=float)
parse.add_argument("--dropout", default=0.2, type=float)
parse.add_argument("--num_epoch", default=1, type=int)
parse.add_argument("--max_vocab_size", default=50000, type=int)
parse.add_argument("--embed_size", default=300, type=int)
parse.add_argument("--enc_hidden_size", default=512, type=int)
parse.add_argument("--num_layers", default=2, type=int)
parse.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
parse.add_argument("--GRAD_CLIP", default=1, type=float)
args = parse.parse_args()

def main():
    #判断gpu是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    #设定随机数种子
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)
    #构建数据处理器
    processor = DataProcessor(args)
    #实例化编码器
    encoder = Encoder(processor.en_tokenizer.vocab_size, args.embed_size,
                      args.enc_hidden_size, args.num_layers, args.dropout)
    #实例化解码器
    decoder = AttnDecoder(processor.cn_tokenizer.vocab_size, args.embed_size,
                          args.enc_hidden_size, args.num_layers, args.dropout)
    #实例化Seq2Seq模型
    model = Seq2Seq(encoder, decoder, args.device, basic_dict = basic_dict)
    model.to(device)

    if args.do_train or args.do_test:
        #加载训练集、验证集和测试集数据
        train_data = processor.get_examples("train")  # 训练集处理并构建批次数据
        eval_data = processor.get_examples("dev")  # 验证集处理并构造批次数据
        test_data = processor.get_examples("test")  # 测试集处理并构造批次数据
    if args.do_train:
        # 判断模型文件存在则加载模型
        if os.path.exists("translate-best.th"):
            model.load_state_dict(torch.load("translate-best.th", map_location=torch.device(device)))

        t_total = args.num_epoch * len(train_data) #计算总训练步数
        # 实例化优化器
        optimizer = AdamW(model.parameters(), lr = args.learning_rate, eps = 1e-8)
        # 实例化学习率衰减器
        scheduler = get_linear_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = args.warmup_steps,
                                                    num_training_steps = t_total)
        best_valid_loss = float('inf')
        for epoch in range(args.num_epoch):
            start_time = time.time()
            #完成一个epoch训练
            train_loss = train(model, train_data, optimizer, scheduler, args.GRAD_CLIP)
            #验证集验证
            valid_loss = evaluate(model, eval_data)
            end_time = time.time()
            if valid_loss < best_valid_loss: #判断验证集损失小于最小损失则保存模型
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), "translate-best.th")
            torch.save(model.state_dict(), "translate-best2.th")
            # 输出训练信息
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
    #判断是否做测试
    if args.do_test:
        model.load_state_dict(torch.load("translate-best.th",
                                         map_location = torch.device(device))) #加载模型参数
        test(model, test_data) #完成模型测试
    #判断是否进行翻译预测
    if args.do_translate:
        model.load_state_dict(torch.load("translate-best2.th", map_location = torch.device(device))) #加载模型参数
        while True:
            title = input("请输入要翻译的英文短句：\n")
            if len(title.strip()) == 0:
                continue
            result = translate(model, title, processor) #进行翻译
            print("翻译后的句子为：", result) #打印结果

if __name__ == "__main__":
    main()