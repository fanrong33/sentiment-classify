# encoding: utf-8
""" 预测
"""

import os
import numpy as np
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

from gensim.models import Word2Vec

from datasets import HOTEL
from models import RNN_EMBED
from datasets import utils




# Hyper Parameters
TIME_STEP = 150
INPUT_SIZE = 400
HIDDEN_SIZE = 50
NUM_LAYERS = 1
NUM_CLASSES = 2

base_dir = os.path.dirname(__file__)


net = RNN_EMBED(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)


net_params_file = os.path.join(base_dir, 'saves/hotel_rnn_params.pkl')
if os.path.isfile(net_params_file):
    net.load_state_dict(torch.load(net_params_file))




sentence = '房间还算干净，但房间内空气不好，有股空调里散发出的霉味；设施极其一般，淋浴龙头拧了半天也不知道倒底哪个是冷水，哪个是热水，到最后差点都关不掉了，估计冷热水龙头大概是装反掉了的；餐厅服务很一般，早餐质量极其一般，餐厅像大单位里的食堂。 '
sentence = '火车晚点,临时改主意去住店,也没有刻意去选择.不过衢州饭店还不错,就是凌晨checkin,前台也要算前一天,收全天房费,说携程预订就是这样,睡了几个小时赶火车,还花了一天的钱.  '
sentence = '在衢州能住到这样的酒店算是很不错了，房间液晶电视，窗口看出去是条小河，还送了水果，算是很不错的啦，就是周围环境一般，没什么可以逛的...'
# sentence = '已多次入住广州文华东方酒店，服务水准一直保持！ 30号入住广州文华东方，31号出差到香港，入住九龙香*里*，后者是非常糟糕的入住体验，非常糟糕！ 经过对比，我才第一次写点评，希望文华一直保持这样卓越的服务！ 因为愉快的入住体验，对于培养客户忠诚度，实在太重要！ 感谢你们让我在广州渡过愉快的一晚！'
# sentence = '设施太老旧了，连瓶水都没有，要自己烧自来水喝的。拖鞋有非常刺鼻的味道，床又小床垫又硬，浴缸还有污迹，电梯里还一股烟味。体验非常不好，住得很不开心。'
sentence = '第一次住如家的酒店，空气很不好，暖气很难受，不过，毕竟价格摆在那里'
sentence = utils.clean_character_sentence(sentence)
stopwords = [w.strip() for w in codecs.open('stopwords.txt', 'r', encoding='utf-8').readlines()]
sentence = utils.segment_character(sentence, stopwords)


# load word2vec model
# word_vectors = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
# word_vectors = Word2Vec.load('../wiki_zh_word2vec/wiki.zh.text.model')
''' Word2Vec(vocab=774118, size=400, alpha=0.025) '''


# inputs = utils.build_words_to_vectors([sentence], TIME_STEP, word_vectors)
# np.save('inputs', inputs)
inputs = np.load(os.path.join(base_dir, 'inputs.npy'))

sentences = Variable(torch.FloatTensor(inputs))
prediction = net(sentences)
print(prediction.data)
_, predicted = torch.max(prediction.data, 1)
print(predicted)
exit()


