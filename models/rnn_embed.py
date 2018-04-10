import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


# 定义神经网络模型
# 定义网络时，需要继承nn.Module，并实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中。
# 如果某一层(如ReLU)不具有可学习的参数，则，既可以放在构造函数中，也可以不放，但建议不放在其中，而在forward中使用nn.functional代替。
""" RNN 模型
1. (input0, state0) -> LSTM -> (output0, state1);
2. (input1, state1) -> LSTM -> (output1, state2);
3. …
4. (inputN, stateN)-> LSTM -> (outputN, stateN+1);
5. outputN -> Linear -> prediction. 通过LSTM分析每一时刻的值, 并且将这一时刻和前面时刻的理解合并在一起, 生成当前时刻对前面数据的理解或记忆. 传递这种理解给下一时刻分析.
"""
# RNN Model (Many-to-One)
class RNN_EMBED(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        """
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(RNN_EMBED, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
        ''' 64x250x50 '''

        # 此处接受的为句子编号序列，长度都已经一致。通过词嵌入成相同维度的矩阵

        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        # out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全0的 state
        batch_size = input.size(0)

        # Set initial states
        h0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))

        # Forward propagate RNN
        output, (hn, cn) = self.lstm(input, (h0, c0))

        # Decode hidden state of last time step
        # 选取最后一个时间点的 out 输出
        # 这里 out[:, -1, :] 的值也是 h_n 的值
        output = output[:, -1, :]
        output = self.fc(output)
        class_prob = F.sigmoid(output)
        return class_prob


        