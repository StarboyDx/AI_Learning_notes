import nltk
import jieba
import torch
from collections import Counter
import os
import numpy as np
#定义特殊字符
basic_dict={'<UNK>':0,"<PAD>":1,'<BOS>':2, '<EOS>':3}
nltk.download('punkt')#下载nltk分词字典
nltk.download('punkt_tab')

#加载文本并分词
def load_file(path):
    en = []
    cn = []
    with open(path, 'r', encoding = "utf-8") as f:
        for line in f.readlines(): # 遍历数据
            line = line.strip().split('\t') # 中英文之间tab分割
            en.append(nltk.word_tokenize(line[0].lower())) #英文分词
            cn.append(jieba.lcut(line[1]))  #中文分词
        return en, cn

#构建词表
def build_tokenizer(sentences, max_vocab_size):
    word_count = Counter() #统计词频
    for sen in sentences: #遍历句子
        for word in sen: #遍历词
            word_count[word] += 1 #统计词频
    ls = word_count.most_common(max_vocab_size) #取出词频最大的max_vocab_size个词
    # 构造词到id的映射字典(注意：前四个idx在basic_dict中被占用了)
    word2idx = {word: idx+4 for idx, (word, _) in enumerate(ls)}
    word2idx.update(basic_dict) # 添加特殊字符

    id2word = {v:k for k, v in word2idx.items()} # 顺便构建反向映射idx > 词
    total_vocab = len(ls) + 4

    return word2idx, id2word, total_vocab # 返回词典

#Tokenizer类负责保存双向映射
class Tokenizer(object):
    def __init__(self, word2idx, id2word, vocab_size):
        self.word2idx = word2idx
        self.id2word = id2word
        self.vocab_size = vocab_size

#实现数据批次迭代
class DataProcessor(object):
    def __init__(self, args):
        self.args = args
        #构造保存源语言字典和目标语言字典的路径
        cached_en_tokenizer = os.path.join(self.args.data_dir, "cached_{}".format("en_tokenizer"))
        cached_cn_tokenizer = os.path.join(self.args.data_dir, "cached_{}".format("cn_tokenizer"))
        # 判断是否已经存在保存的字典  如果存在就直接读取，不存在则需要加载训练集数据进行构建
        if not os.path.exists(cached_en_tokenizer) or not os.path.exists(cached_cn_tokenizer):
            en_sents, cn_sents = load_file(self.args.data_dir + "train.txt") #加载训练集并进行分词
            en_word2idx, en_id2word, en_vocab_size = build_tokenizer(en_sents, self.args.max_vocab_size) #使用英文分词结果构造英文词典
            cn_word2idx, cn_id2word, cn_vocab_size = build_tokenizer(cn_sents,
                                                                     self.args.max_vocab_size)  # 使用中文分词结果构造中文词典
            torch.save([en_word2idx, en_id2word, en_vocab_size], cached_en_tokenizer)  # 保存英文词典
            torch.save([cn_word2idx, cn_id2word, cn_vocab_size], cached_cn_tokenizer)  # 保存中文词典
        else:
            en_word2idx, en_id2word, en_vocab_size = torch.load(cached_en_tokenizer)  # 加载英文词典
            cn_word2idx, cn_id2word, cn_vocab_size = torch.load(cached_cn_tokenizer)  # 加载中文词典
        # 构建源语言映射器和目标语言映射器
        self.en_tokenizer = Tokenizer(en_word2idx, en_id2word, en_vocab_size)  # 使用词典构造英文映射器
        self.cn_tokenizer = Tokenizer(cn_word2idx, cn_id2word, cn_vocab_size)  # 使用词典构造中文映射器

    def tokenize2id(self, en_sentences, cn_sentences, sort_reverse = True):
        out_en_sents = [[self.en_tokenizer.word2idx.get(word, basic_dict['<UNK>']) for word in sen] + [basic_dict['<EOS>']]
                        for sen in en_sentences] # 将英文词序列转换成id序列
        out_cn_sents = [[self.cn_tokenizer.word2idx.get(word, basic_dict['<UNK>']) for word in sen] + [basic_dict['<EOS>']]
                        for sen in cn_sentences] # 将中文词序列转换成id序列
        if sort_reverse:
            # 按源语言句子长度降序排序  得到排序后的下标
            sorted_index  = sorted(range(len(out_en_sents)), key = lambda x: len(out_en_sents[x]), reverse = True)
            # 根据下标排序
            out_en_sents = [out_en_sents[idx] for idx in sorted_index]
            out_cn_sents = [out_cn_sents[idx] for idx in sorted_index]
        return out_en_sents, out_cn_sents

    def getminibatches(self, n, batch_size, shuffle = True):
        minibatches = np.arange(0, n, batch_size) #获取起始下标
        if shuffle:
            np.random.shuffle(minibatches)
            # 每批次内的数据打乱

        result = []
        for idx in minibatches: #产生批次数据
            result.append(np.arange(idx, min(n, idx + batch_size))) #截取一个批次数据
        return result

    def prepare_data(self, seqs):
        # 处理每个batch句子（一个batch中句子长度可能不一致，需要pad）
        batch_size = len(seqs)
        lengthes = [len(seq) for seq in seqs] #每个句子的长度列表
        max_length = max(lengthes)
        # 初始化句子矩阵都为<PAD>的id
        x = np.ones((batch_size, max_length)).astype('int32')
        x = x * basic_dict['<PAD>']
        for idx in range(batch_size):
            # 按行将每行句子赋值进去
            x[idx, :lengthes[idx]] = seqs[idx]
        #句子真实长度
        x_lengths = np.array(lengthes).astype('int32')
        return x, x_lengths

    def get_examples(self, set_type):
        #读取数据并分词
        en_sents, cn_sents = load_file(os.path.join(self.args.data_dir, set_type+".txt"))
        # 将词转换成id
        out_en_sents, out_cn_sents = self.tokenize2id(en_sents, cn_sents)
        # 数据分成批次
        batch_size = self.args.batch_size if set_type != 'test' else 1
        minibatches = self.getminibatches(len(out_en_sents), batch_size)
        # 遍历第一个批次 进行填充并转换为tensor
        all_examples = []
        for minibatch in minibatches: # 遍历每一个批次的数据
            mb_en_sentences = [out_en_sents[i] for i in minibatch]
            mb_cn_sentences = [out_cn_sents[i] for i in minibatch]

            mb_x, mb_x_len = self.prepare_data(mb_en_sentences) #对短句子进行pad
            mb_y, mb_y_len = self.prepare_data(mb_cn_sentences)
            # 数据转换为tensor
            mb_x = torch.from_numpy(mb_x).to(self.args.device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(self.args.device).long()

            mb_y = torch.from_numpy(mb_y).to(self.args.device).long()
            mb_y_len = torch.from_numpy(mb_y_len).to(self.args.device).long()

            all_examples.append((mb_x, mb_x_len, mb_y, mb_y_len))

        return all_examples