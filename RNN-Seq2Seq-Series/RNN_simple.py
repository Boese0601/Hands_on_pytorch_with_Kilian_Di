import sys
sys.path.append('./RNN.py')
from RNN import *
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

# class TransformerModel(nn.Module):
#     def __init__(self,transformer_layer,vocab_size):
#         self.transformer = transformer_layer
#         self.encoder_layers = 6
#         self.decoder_layers = 6
#         self.nheads = vocab_size
#
#     def forward(self,x):

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vocab_size = Vocabulary size type:int
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    num_hiddens = 256
    # rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens) # 已测试

    # 输入形状为(时间步数, 批量大小, 输入个数)。其中输入个数即one-hot向量长度（词典大小）
    # 其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输入。需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2,bidirectional=False)

    num_steps = 35
    batch_size = 16
    state = None
    model = RNNModel(rnn_layer,vocab_size).to(device)

    num_epochs, batch_size, lr, clipping_theta = 2500, 128, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)

    print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

    # X = torch.rand(num_steps, batch_size, vocab_size)
    # Z = torch.rand(num_steps, batch_size, vocab_size)
    # print(X.shape,Z.shape)
    # Y, state_new = rnn_layer(X, state)
    # print(Y.shape, state_new.size())
    # Transformerlayer = nn.Transformer(nhead=vocab_size,num_decoder_layers=6,num_encoder_layers=6,d_model=vocab_size)
    # Y = Transformerlayer(X,Z)
    # print(Y.shape)



