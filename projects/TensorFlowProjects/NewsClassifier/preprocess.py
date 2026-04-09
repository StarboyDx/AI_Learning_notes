from config import Config #引入配置类
from tqdm import tqdm #进度条
import numpy as np
import random #用来实现随机打乱
UNK, PAD= '<UNK>', '<PAD>'  # 未知字，padding符号,数字

def get_vocab():
    """
    读取分词文档
    :return:分词数组与各分词索引的字典
    """
    vocab_path = Config().get("data_path", "vocab_path")#获取词典文件路径
    with open(vocab_path, "r", encoding="utf-8") as f:
        infile = f.readlines()
    id2w = [PAD,UNK]+list([word.replace("\n", "") for word in infile])#加入pad和未知词 可以根据id查到词
    w2id = dict(zip(id2w, range(len(id2w))))#转换为word2id字典
    return id2w, w2id

class DatasetIterater(object):
    def __init__(self,vocab,mode='train',onehot=True,shuffle=True):
        self.config = Config()#获取配置信息
        self.mode=mode#模型类型
        self.vocab =vocab#词典
        self.onehot = onehot#是否对label进行onehot处理
        self.shuffle=shuffle#是否打乱
        self.batch_szie=self.config.get("training_rule","batch_size")#批次
        self.class_eye = np.eye(len(self.config.get('category', "category")))  # 用来对label进行onehot
        self.classes=self.config.get('category',"cat2id")#类别
        self.max_length=self.config.get('training_rule',"seq_length")#最大长度
        self.index = 0#迭代下标

        self.batchs=self.get_batchs()
        if self.shuffle:#训练数据必须打乱顺 不打乱数据模型无法训练
            random.shuffle(self.batchs)
        self.n_batchs=len(self.batchs)#批数


    #加载数据函数
    def load_dataset(self):
        contents = []
        #根据mode获取对应的数据集路径（train、val、test）
        path=self.config.get("data_path",self.mode)
        print("正在加载：",self.mode)
        #读取文件
        with open(path, 'r', encoding='UTF-8') as f:
            #遍历每一行
            for line in tqdm(f.readlines()):
                lin = line.strip()#去除两边空格
                if not lin:
                    continue
                #获取文本内容和标签
                label, content = lin.split('\t')#分割
                words_line=[]
                # 将词转换成对应的id 并截取最长长度
                for i,word in enumerate(content):
                    if i>=self.max_length:#超过最大长度不再处理
                        break
                    words_line.append(self.vocab.get(word, self.vocab.get(UNK)))#词转换成对应的id 未登录词转为UNK
                #返回每句话的id序列，标签和真实长度
                contents.append((words_line, self.classes[label]))
        return contents  # [([...], 0), ([...], 1), ...]
    def get_batchs(self):
        datas=self.load_dataset()#读取数据
        self.num_samples=len(datas)
        datas=sorted(datas,key=lambda x:len(x[0]))#按照序列长度降序排序
        batchs=[]
        idxs=list(range(0,self.num_samples,self.batch_szie))+[-1]#构建批次分割下标
        for i in range(len(idxs)-1):
            batch=datas[idxs[i]:idxs[i+1]]#截取
            lens=[len(s[0]) for s in batch]#计算长度
            max_len=max(lens)#获取该批次最大长度
            sentences=np.array([[self.vocab.get(PAD)]*(max_len-len(s[0]))+s[0] for s in batch])#在前面填充
            labels=np.array([s[1] for s in batch])#转成array
            if self.onehot:
                labels=self.class_eye[labels]#对label进行onehot处理
            batchs.append((sentences,labels))
        return batchs

    def __next__(self):
        #index超过了 表示一个轮次完了
        if self.index >= self.n_batchs:
            self.index = 0
            if self.shuffle:#每个轮次后打乱数据
                random.shuffle(self.batchs)
            raise StopIteration
        else:
            batchs=self.batchs[self.index]
            self.index+=1#下标+1
            return batchs#返回一批数据


    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batchs


def preprocess(sentences):
    """
    推理时对文本进行预处理
    :param sentences: 句子列表
    :return: 转换成id后的数组
    """
    id2w, w2id = get_vocab()
    max_length = Config().get("training_rule", "seq_length")#最大序列长度
    contents = []
    # 遍历每一行
    for line in sentences:#遍历所有句子
        line = line.strip()
        words_line = []
        # 将词转换成对应的id 并截取最长长度
        for i, word in enumerate(line):
            if i >= max_length:
                break#超过长度就退出循环
            words_line.append(w2id.get(word, w2id.get(UNK)))#词转id
        # 返回每句话的id序列，标签和真实长度
        contents.append(words_line)

    # 进行填充pad
    maxlen = max([len(s) for s in contents])
    pad = w2id.get(PAD)#获取PAD的id
    contents = [[pad] * (maxlen - len(s)) + s for s in contents]#pad处理
    return np.array(contents)


if __name__=='__main__':
    _, vocab = get_vocab() # 获取词表
    print(vocab) # 打印词表
    test = DatasetIterater(vocab) # 创建数据集
    for x, y in test: # 遍历数据
        print("x\n", x) # 打印数据
        print("y:\n", y)
        break