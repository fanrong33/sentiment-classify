# encoding: utf-8
""" 神经网络模型
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class ConvLSTMSa(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, num_classes):
        super(ConvLSTMSa, self).__init__()

        self.input_size = input_size   # embedding_dim
        self.hidden_size = hidden_size # hidden_dim
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(vocab_size, input_size)

        # TODO 待优化为传入参数
        word_vectors = np.load('data/wordVectors.npy')
        print('Loaded the word vectors')
        self.word_embedding.weight.data = torch.from_numpy(word_vectors)


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=(2, hidden_size), stride=(1, 1), padding=(0, 0))
        
        self.lstm = nn.LSTM(    # LSTM 效果要比 nn.RNN() 好多了
            input_size=input_size,      # 图片每行的数据像素点
            hidden_size=hidden_size,    # rnn hidden unit
            num_layers=num_layers,      # 有几层 RNN layers
            # dropout=0.2,
            batch_first=True,           # input & output 会是以 batch size 为第一维度的特征集  e.g. (batch, time_step, input_size)
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size, num_classes)  # 输出为全连接层

    def forward(self, input):
        batch_size = input.size(0)
        time_step = input.size(1)
        ''' 1x61 '''
        # 此处接受的为句子编号序列，长度都已经一致。通过词嵌入成相同维度的矩阵
        embeds = self.word_embedding(input)
        ''' 1x61x50 '''
        in_embed = embeds.view(batch_size, 1, time_step, -1)
        ''' 1x1x61x50 '''
        ls_inputs = self.conv1(in_embed)  # ls_inputs的维度为：embedding_dim*1*len(sentence)
        ''' 1x50x60x1 '''


        # 维度转化
        ls_inputs2 = torch.transpose(ls_inputs, 1, 3)
        ''' 1x1x60x50 '''
        ls_inputs3 = ls_inputs2.contiguous().view(batch_size, 249, -1) # 60
        ''' 1x60x50 '''
        
        # ls_inputs3 = torch.transpose(ls_inputs2, 1, 2)
        # [torch.FloatTensor of size 1x60x1x50]

        # Set initial states
        h0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))

        # LSTM输入，想想评论的解构 batch, time_step, input_size
        lstm_out, lstm_hidden = self.lstm(ls_inputs3, (h0, c0))
        ''' 1x60x50 '''

        tag_space = self.fc(lstm_out[:,-1,:])
        # print(tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print(tag_scores)
        return tag_scores


