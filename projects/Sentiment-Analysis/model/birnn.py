import torch
from torch import nn

class BiRNN(nn.Module):
    def __init__(self, embedding_matrix, num_hiddens=100, num_layers=1):
        super(BiRNN, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        #嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        #加载词向量权重
        self.embedding.weight.data.copy_(embedding_matrix)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.encoder(embeddings)
        encoding = torch.cat((output[:, 0, :], output[:, -1, :]), dim=1)
        outs = self.decoder(encoding)
        return outs