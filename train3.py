# encoding: utf-8
""" 训练神经网络
"""

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from models import ConvLSTMSa
from datasets import MR
from datasets import utils

torch.manual_seed(1)


# Hyper Parameters
TIME_STEP = 61
INPUT_SIZE = 50
HIDDEN_SIZE = 50
NUM_LAYERS = 1
NUM_CLASSES = 2

LR = 1e-3
BATCH_SIZE = 64
EPOCH = 100       # 训练


# Load Datasets
train_dataset = MR(root='data/MR/', train=True)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


test_dataset = MR(root='data/MR/', train=False)
print(len(test_dataset))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

(test_sents, test_labels) = iter(test_loader).next()



word2idx = np.load('data/word2idx.npy').tolist()


net = ConvLSTMSa(len(word2idx), INPUT_SIZE,  HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
print(net)


# if os.path.isfile('saves/mr_convlstmsa_params.pkl'):
#     net.load_state_dict(torch.load('saves/mr_convlstmsa_params.pkl'))


# optimizer & Loss
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()  NLLLoss


# Train Model
for epoch in range(EPOCH):
    for step, (sents, labels) in enumerate(train_loader):

        sentences = utils.prepare_sequence(sents, word2idx, TIME_STEP)
        sentences = Variable(torch.LongTensor(sentences))
        labels = Variable(torch.LongTensor(labels))
        # 这里词嵌入在神经网络内创建，推荐跟图片卷积一样，输入为1x28x28，保持结构的一致
        # 将词嵌入部分解构为独立部分，预先处理
        # no, 事实上，input 为 文章ids序列，反而更适合，因为内部模型可以是任何的组合重装方式
        # conv => batch size x 1 x 61 x 50
        # lstm => batch size x time_step x input_size

        optimizer.zero_grad()

        prediction = net(sentences)
        
        loss = loss_func(prediction, labels)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            sentence_ins = utils.prepare_sequence(test_sents, word2idx, TIME_STEP)
            sentence_ins = Variable(torch.LongTensor(sentence_ins))
            test_prediction = net(sentence_ins)
            _, predicted = torch.max(test_prediction.data, 1)
            test_accuracy = sum(predicted == test_labels)/float(test_labels.size(0))

            print('Epoch: %d, Step: %d, Training Loss: %.4f, Test Accuracy: %.3f' % 
                (epoch, step, loss.data[0], test_accuracy))


            torch.save(net.state_dict(), 'saves/mr_convlstmsa_params.pkl')  # 只保存网络中的参数（速度快，占内存少）


