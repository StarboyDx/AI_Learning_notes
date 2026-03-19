import torch

class BasicConfigs():
    #数据存放参数
    neg = 'Dataset/neg' #负样本目录
    pos = 'Dataset/pos' #正样本目录
    data_path = 'Dataset' #分割后数据存放目录
    text_vocab_path = 'model_storage/text.vocab'#文本词典存放目录
    label_vocab_path ='model_storage/label.vocab'#标签词典存放目录
    stop_word_path = 'Dataset/stopword.txt' #停用词文件路径
    # 词向量参数
    embedding_loc = 'Dataset/sgns.wiki.word' #词向量参数
    # 模型训练参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #设备
    lr = 0.001
    dropout_rate = 0.5 #随机失活比例
    train_embedding = True #是否训练词嵌入向量
    batch_size = 64 #批次大小
    alpha = 0.001 #L2惩罚项系数
    # bilstm配置参数
    num_hiddens = 100 #lstm神经元数
    num_layers = 1 #lstm层数
    save_model_dir = {
        'birnn':'model_storage/model_rnn.pt',
    }