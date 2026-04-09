from preprocess import DatasetIterater, get_vocab
from model import TextCNN, TextLSTM

if __name__ == '__main__':
    id2w, w2id = get_vocab()  # 获取词表
    train_data = DatasetIterater(w2id)  # 构建训练集 打乱批次
    val_data = DatasetIterater(mode='val', vocab=w2id, shuffle=False)  # 构建验证集 不打乱
    test_data = DatasetIterater(mode='test', vocab=w2id, shuffle=False)  # 构建测试集 不打乱
    # CNN_model = TextCNN(vocab_size=len(id2w))  # 实例化TextCNN模型
    # CNN_model.train(train_data, val_data)  # 模型训练
    # CNN_model.test(test_data)

    LSTM_model = TextLSTM(vocab_size=len(id2w))
    LSTM_model.train(train_data, val_data)
    LSTM_model.test(test_data)
