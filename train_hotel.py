# encoding: utf-8
""" 训练神经网络
"""

import os
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision

from models import RNN_EMBED
from datasets import HOTEL
from datasets import utils

torch.manual_seed(1)


# Hyper Parameters
TIME_STEP = 150
INPUT_SIZE = 400
HIDDEN_SIZE = 50
NUM_LAYERS = 1
NUM_CLASSES = 2

LR = 1e-3
BATCH_SIZE = 64
EPOCH = 100   # 训练



# Load Datasets
train_dataset = HOTEL(root='data/hotel/', train=True)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_dataset = HOTEL(root='data/hotel/', train=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=2)

(test_data, test_labels) = iter(test_loader).next()


# Construct Model
net = RNN_EMBED(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
print(net)


if os.path.isfile('saves/hotel_rnn_params.pkl'):
    net.load_state_dict(torch.load('saves/hotel_rnn_params.pkl'))



# Optimizer & Loss
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# Train Model
for epoch in range(EPOCH):
    for step, (sentences, labels) in enumerate(train_loader):
        
        sentences = Variable(sentences.float())
        labels = Variable(labels.long())


        optimizer.zero_grad()

        prediction = net(sentences)
        loss = loss_func(prediction, labels)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            sentences = Variable(test_data.float())
            test_prediction = net(sentences)

            _, predicted = torch.max(test_prediction.data, 1)
            test_accuracy = sum(predicted == test_labels.long())/float(test_labels.size(0))

            print('Epoch: %d, Step: %d, Training Loss: %.4f, Test Accuracy: %.3f' % 
                (epoch, step, loss.data[0], test_accuracy))

            torch.save(net.state_dict(), 'saves/hotel_rnn_params.pkl')  # 只保存网络中的参数（速度快，占内存少）


