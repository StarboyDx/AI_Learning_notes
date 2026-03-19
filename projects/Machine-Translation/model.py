import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout = 0.5, bidireactional = True):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim #隐藏层维度
        self.n_layers = n_layers #GRU层数
        self.embedding = nn.Embedding(vocab_size, emb_dim) #嵌入层
        #循环神经网络层
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout = dropout, batch_first = True,
                          bidirectional = bidireactional)

    def forward(self, input_seqs, input_lengths, hidden):
        # 输入input_seqs = [batch, seq_len]
        embedded = self.embedding(input_seqs) #进行词嵌入
        # 嵌入结果embedded = [batch, seq_len, embed_dim]
        # 对序列进行长度排序并pack压缩
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first = True, enforce_sorted = False)
        outputs, hidden = self.gru(packed, hidden) #进行gru计算
        #对计算结果进行解压
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True, total_length = max(input_lengths))
        # 输出outputs = [batch，seq_len,hid_dim * n directions]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, hid_dim):
        super(Attn, self).__init__()
        # 注意力参数W
        # tip: transformer直接QK点乘，这里QK拼接送到一个Linear匹配
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        # 注意力参数V
        self.v = nn.Parameter(torch.randn(hid_dim))

    def forward(self, hidden, encoder_outputs):
        # 实现公式计算
        # hidden是一个时刻的(一个词)[32, 1, 512]，encoder_outputs是10个词[32, 10, 1024]，
        # .expand复制然后在dim = 2上拼接成[32, 10, 1536]，然后送入线性层再激活
        energy = self.attn(torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)).tanh()
        #广播写法：先让参数向量 v [512] 和 energy [32, 10, 512] 逐元素相乘，然后用 torch.sum(..., dim=2) 把最后那个 512 维全部加起来。
        attn_energies = torch.sum(self.v * energy, dim = 2)
        # 按句子长度的维度（dim=1）做了一次 Softmax
        # .unsqueeze(1) 在中间硬插了一个维度，把 [32, 10] 变成了 [32, 1, 10]
        # 为后面bmm批量矩阵乘法准备（bmm不管第一维，[1,10] X [10, 1024]，10消掉），得到[32, 1, 1024]的上下文信息
        return F.softmax(attn_energies, dim = 1).unsqueeze(1)

class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers = 1, dropout = 0.5):
        super(AttnDecoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim #隐藏神经元数
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)
        #gru解码层
        self.gru = nn.GRU(self.hid_dim, self.hid_dim, n_layers,
                          dropout = (0 if n_layers == 1 else dropout),
                          batch_first = True)
        #注意力层
        self.att = Attn(hid_dim)
        #输入映射层
        self.concat = nn.Linear(hid_dim * 2 + emb_dim, hid_dim)
        #输出映射层
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, seq_in, state, encoder_outputs):
        embedded = self.embedding(seq_in) #词嵌入
        # 取出第一层的隐藏状态, GRU 可以叠很多层（比如 2 层、3 层）。在这里，作者选择只抽出最底层的隐藏状态来作为 Query
        # unsqueeze把取出的[32, 512]变成[32, 1, 512]
        onelayerhidden = state[0, :, :].unsqueeze(1) # batchsize,layer,hiddensize
        # Query ([32, 1, 512]) 查到了注意力权重 ([32, 1, 10])
        atten_weights = self.att(onelayerhidden, encoder_outputs)
        # bmm加权求和, 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = atten_weights.bmm(encoder_outputs)
        # 拼接编码向量和解码器输入的嵌入向量
        concat_input = torch.cat((embedded, context), 2)
        # 输入映射，把上一步拼接的1324维信息，重新映射成512维，然后激活
        input = torch.tanh(self.concat(concat_input))
        # 解码
        output, hidden = self.gru(input, state)
        # 输出映射
        output = self.out(output.squeeze(1)) #[32, 1, 512] -- [32, 512]
        output = F.log_softmax(output, dim = 1)
        return output, hidden, atten_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, predict = False, basic_dict = None, max_len = 100):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        # 状态映射层  编码器双向  解码器单向
        self.first_state = nn.Linear(self.encoder.hid_dim * 2, self.decoder.hid_dim)
        self.predict = predict
        self.basic_dict = basic_dict # decoder的字典，存放特殊token对应的id
        self.max_len = max_len # 翻译时最大输出长度

    def forward(self, input_batches, input_lengths,
                target_batches = None, target_lengths = None,
                teacher_forcing_ratio = 0.5):
        batch_size = input_batches.shape[0] #取出批次大小
        # 取出特殊字符
        BOS_token, EOS_token, PAD_token = self.basic_dict["<BOS>"], self.basic_dict["<EOS>"], self.basic_dict["<PAD>"]
        enc_n_layers = self.encoder.gru.num_layers
        enc_n_directions = 2 if self.encoder.gru.bidirectional else 1 #编码器的方向
        # 初始化编码器初始状态
        encoder_hidden = torch.zeros(enc_n_layers * enc_n_directions,
                                     batch_size, self.encoder.hid_dim, device = self.device)
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, encoder_hidden) #编码器计算
        # 初始化解码器第一个时刻输入，都是BOS
        decoder_input = torch.tensor([BOS_token] * batch_size,
                                      dtype = torch.long, device = self.device).view(batch_size, -1)
        # 将encoder_hidden进行reshape后transpose，将enc_n_directions和hid_dim放到最后两个维度
        encoder_hidden = encoder_hidden.reshape(enc_n_layers, enc_n_directions, batch_size, -1).transpose(1, 2)
        # 将encoder_hidden进行reshape，变为enc_n_layers,batch_size,enc_n_directions*hid_dim
        encoder_hidden = encoder_hidden.reshape(enc_n_layers, batch_size, -1)
        # 进行映射得到shape为[n_layers, batch, hid_dim]的初始状态
        decoder_hidden = self.first_state(encoder_hidden).tanh()
        #预测阶段没有真实序列，不能用Teacher forcing技术
        #使用简单的贪心算法来寻找局部最优路径，取最大值作为预测值喂给下一时刻
        if self.predict:
            #一次只输入一句话
            assert batch_size == 1, "batch_size of predict phase must be 1!"
            output_tokens = []
            while True:
                #解码器解码一次
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.topk(1) #取出当前时刻的预测wordid和下标
                decoder_input = topi.detach()
                output_token = topi.squeeze().detach().item()
                # 判断预测wordid是否为终止标识或者预测序列长度已经超过最大值 满足条件的话需要停止推理
                if output_token == EOS_token or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token) #输出结果保存到列表
            return output_tokens

        else:
            max_target_length = max(target_lengths) #计算序列的最大长度
            #初始化解码输出为0
            all_decoder_outputs = torch.zeros((max_target_length, batch_size,
                                              self.decoder.output_dim), device = self.device)
            #遍历最长序列的所有时刻
            for t in range(max_target_length):
                # 将上一时刻的隐状态decoder_hidden、当前时刻的输入decoder_input和编码器的所有时
                # 刻的输出encoder_outputs送入解码器解码一次，
                # 得到当前时刻的输出decoder_output和当前时刻的隐状态decoder_hidden
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                all_decoder_outputs[t] = decoder_output # 保存当前时刻输出
                # 随机数判断本时刻是否使用teacher_forcing技术
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    decoder_input = target_batches[:, t].view(-1, 1) # 下一个输入来自训练真实数据
                else:
                    topv, topi = decoder_output.topk(1) # 拿到最大的值及其下标
                    decoder_input = topi.detach()  # 下一个输入来自解码器上一个时刻的预测值

            loss_fn = nn.NLLLoss(ignore_index = PAD_token) # 定义损失函数
            #计算损失值
            loss = loss_fn(
                all_decoder_outputs.reshape(-1, self.decoder.output_dim),  # [seq_len*batch, output_dim]
                target_batches.transpose(1, 0).reshape(-1)  # [seq_len*batch]
            )
            return loss
