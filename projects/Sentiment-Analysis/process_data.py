import pandas as pd
from sklearn.model_selection import train_test_split
from configs import BasicConfigs
from glob import glob #用来遍历文件夹中的所有文件
import jieba #分词
import pickle #用于读取和存储二进制（词典）文件
#torchtext：用来构建训练字段、词向量、词典和批次迭代器
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re #正则匹配，负责清洗文本
import torch

bc = BasicConfigs()

# 原生python实现词表类
class Vocab:
    def __init__(self, tokens, min_freq=1, specials=['<pad>', '<unk>']):
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        # 统计词频
        counter = Counter(tokens)
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

        self.pad_index = self.stoi['<pad>']
        self.unk_index = self.stoi['<unk>']

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        # 将单词列表转为索引列表，找不到的词返回 unk_index
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

    def get_itos(self):
        return self.itos

# 还是原生 Python 加载预训练词向量
def load_pretrained_vectors(filepath):
    word2vec = {}
    embed_dim = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # 跳过第一行的 meta 信息 (如 vocab_size embed_dim)
            if len(parts) <= 2:
                continue
            word = parts[0]
            vec = [float(x) for x in parts[1:]]
            if embed_dim is None:
                embed_dim = len(vec)
            word2vec[word] = vec
    return word2vec, embed_dim

def load_data_to_csv():
    # 遍历文件夹 读取数据
    contents = []
    for file in glob(bc.neg + "/*.txt"): #读取neg目录中的所有文件
        with open(file, 'r', encoding = 'utf-8') as f:
            content = ''.join([line.strip() for line in f.readlines()]) #读取文本
            contents.append([content, 'neg'])
    for file in glob(bc.pos + "/*.txt"): #读取pos
        with open(file, 'r', encoding = 'utf-8') as f:
            content = ''.join([line.strip() for line in f.readlines()])
            contents.append([content, 'pos'])
    # 打乱顺序并存储到train.csv,test.csv,val.csv
    # 封装df
    df = pd.DataFrame(contents, columns=['text', 'label'])
    train, test = train_test_split(df, test_size=0.1, random_state=12)  # 数据分割
    train, val = train_test_split(train, test_size=0.2, random_state=12)  # 训练集再分割
    train.to_csv(bc.data_path + '/train.csv', index=False)  # 保存数据
    val.to_csv(bc.data_path + '/val.csv', index=False)
    test.to_csv(bc.data_path + '/test.csv', index=False)
    print("process finished~")

# 文本清洗
def clearTxt(line):
    if line != '':
        line = line.strip()
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line

def my_cut(line):
    line=clearTxt(line)#清洗
    return jieba.lcut(line)#分词并返回中

#自定义Dataset
class TextDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.loc[idx, 'text'], self.df.loc[idx, 'label']

# 辅助构建词表的生成器
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield my_cut(text)

def prepare_vocab(is_train = True):
    if is_train:
        train_dataset = TextDataset(bc.data_path + '/train.csv')
        val_dataset = TextDataset(bc.data_path + '/val.csv')

        #1. 构建词表
        print("Building vocabulary")
        all_tokens = []
        for i in range(len(train_dataset)):
            text, _ = train_dataset[i]
            all_tokens.extend(my_cut(text))
        vocab = Vocab(all_tokens, specials = ['<pad>', '<unk>'])

        #2. 构建标签映射表
        labels = train_dataset.df['label'].unique()
        label_map = {label: idx for idx, label in enumerate(labels)}

        #3. 构建词向量矩阵
        print("Loading pre-trained vectors...")
        word2vec, embed_dim = load_pretrained_vectors(bc.embedding_loc)
        if embed_dim is None:
            embed_dim = 300  # 如果文件为空或读取失败，默认给 300 维

        embedding_matrix = torch.zeros((len(vocab), embed_dim))
        for i, word in enumerate(vocab.get_itos()):
            if word in word2vec:
                embedding_matrix[i] = torch.tensor(word2vec[word])
            else:
                embedding_matrix[i] = torch.randn(embed_dim) * 0.1  # 未知词随机初始化

        #4. 保存模型依赖字典
        with open(bc.text_vocab_path, 'wb') as f:
            pickle.dump({'vocab': vocab, 'embedding_matrix': embedding_matrix }, f)
        with open(bc.label_vocab_path, 'wb') as f:
            pickle.dump(label_map, f)

        #5. 定义DataLoader的Collate Function（动态Padding）
        def collate_batch(batch):
            text_list, label_list = [], []
            for (_text, _label) in batch:
                processed_text = torch.tensor(vocab(my_cut(_text)), dtype=torch.int64)
                text_list.append(processed_text)
                label_list.append(label_map[_label])

            #pad_sequence默认batch_first = False
            text_list = pad_sequence(text_list, padding_value = vocab.pad_index, batch_first=True)
            label_list = torch.tensor(label_list, dtype=torch.int64)
            return text_list, label_list

        train_loader = DataLoader(train_dataset, batch_size=bc.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=bc.batch_size, shuffle=False, collate_fn=collate_batch)

        return vocab, label_map, train_loader, val_loader, embedding_matrix

    else:
        #验证/推理阶段加载
        with open(bc.text_vocab_path, 'rb') as f:
            data = pickle.load(f)
            vocab = data['vocab']
            embedding_matrix = data['embedding_matrix']
        with open(bc.label_vocab_path, 'rb') as f:
            label_map = pickle.load(f)
        return vocab, label_map, embedding_matrix

def transform_data(record, vocab, label_map):
    if not isinstance(record, dict):
        raise ValueError('Make sure data is dict')
    tokens = my_cut(record['text'])
    res = vocab(tokens)
    # 增加batch维度 -> [1, seq_len]
    data = torch.tensor(res, dtype=torch.int64).unsqueeze(0)

    if 'label' in record:
        label = torch.tensor([label_map[record['label']]], dtype=torch.int64)
    else:
        label = None
    return data, label

if __name__ == '__main__':
    load_data_to_csv()