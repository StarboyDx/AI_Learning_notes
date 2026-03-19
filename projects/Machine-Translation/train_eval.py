import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch
import time
import nltk

def epoch_time(start_time, end_time):
    elasped_time = end_time - start_time
    elasped_mins = int(elasped_time / 60)
    elasped_secs = int(elasped_time - (elasped_mins * 60))
    return elasped_mins, elasped_secs

def train(model, data, optimizer,
          scheduler, clip = 1,
          teacher_forcing_ratio = 0.5,
          print_every = 100):
    model.predict = False
    model.train()
    print_loss_total = 0 # 每次打印都重置
    epoch_iteration = tqdm(data, desc = 'iteration')
    epoch_loss = 0
    for i, (input_batchs, input_lens, target_batchs, target_lens) in enumerate(epoch_iteration):
        optimizer.zero_grad()
        #模型计算
        loss = model(input_batchs, input_lens,
                     target_batchs, target_lens,
                     teacher_forcing_ratio)
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        if (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)
    return epoch_loss / len(data)

def evaluate(model, data, print_every = 100):
    model.predict = False
    model.eval()
    print_loss_total = 0
    epoch_loss = 0
    eval_iteration = tqdm(data, desc = 'eval iteration')
    with torch.no_grad():
        for i, (input_batchs, input_lens, target_batchs, target_lens) in enumerate(eval_iteration):
            #模型计算损失
            loss = model(input_batchs, input_lens,
                         target_batchs, target_lens,
                         teacher_forcing_ratio = 0)
            print_loss_total += loss.item()
            epoch_loss += loss.item()
            if (i + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\tCurrent Loss: %.4f' % print_loss_avg)
    return epoch_loss / len(data)

def test(model, data):
    model.predict = True #设置模型为预测模式
    model.eval()
    preds = []
    targets = []
    test_iteration = tqdm(data, desc = 'test iteration')
    with torch.no_grad():
        for input_batchs, input_lens, target_batchs, _ in test_iteration:
            pred_tokens = model(input_batchs, input_lens) #得到一条预测结果
            preds.append(pred_tokens) #保存预测结果
            targets.append([list(target_batchs[0].cpu().numpy())]) #保存真实标签
    bleu_score = corpus_bleu(targets, preds) #计算bleu指标
    print('Corpus BLEU: {}'.format(bleu_score * 100))

def translate(model, sample, processor):
    #分词
    title = nltk.word_tokenize(sample.lower()) + ['<EOS>']
    #映射id
    title_num = [processor.en_tokenizer.word2idx.get(word, 0) for word in title]
    #构造成Tensor
    x = torch.from_numpy(np.array(title_num).reshape(1, -1)).long().to(processor.args.device)
    x_len = torch.from_numpy(np.array([len(title_num)])).long().to(processor.args.device)
    model.predict = True
    model.eval()
    output_tokens = model(x, x_len)
    output_tokens = [processor.cn_tokenizer.id2word[t] for t in output_tokens] #输出结果转换成文本
    return "".join(output_tokens)
