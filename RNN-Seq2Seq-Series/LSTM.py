import sys
sys.path.append('./RNN.py')
sys.path.append('./RNN_simple.py')
from RNN_simple  import *
from RNN import *
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
num_hiddens = 256
lr = 1e-2 # 注意调整学习率

lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size)
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
