# encoding: utf-8
""" 自定义数据集 MR
最原始的数据
@version 1.0.1 build 20180405
"""

import torch
import torch.utils.data as Data
import torchvision

import os
import numpy as np
import codecs
import random

from . import utils


""" 
  预处理 word2idx
"""
def build_word_to_idx():
    # already tokenized and there is no standard split
    # the size follow the Mou et al. 2016 instead
    file_pos = '../data/MR/rt-polarity.pos'  # TODO 路径问题
    file_neg = '../data/MR/rt-polarity.neg'
    print('loading MR datasets from,', file_pos, 'and', file_neg)

    # codecs.open(): 读入数据时，直接解码操作。防止编码格式问题。
    # .read(), .readlines(), .readline() 这些方法区别是：
    # read() 读取整个文件成一个字符串
    # readlines() 读取整个文件，自动将文件分成行的列表，供for line in fp.readlines() 调用
    # readline() 读取一行数据，速度慢。通常在内存不够的情况下使用
    pos_sents = codecs.open(file_pos, 'r', 'utf8').read().split('\n')
    # print(pos_sents[0:2])
    neg_sents = codecs.open(file_neg, 'r', 'utf8').read().split('\n')

    random.seed(1000)
    random.shuffle(pos_sents)
    random.shuffle(neg_sents)

    print(len(pos_sents))
    ''' 5331 '''
    print(len(neg_sents))
    ''' 5331 '''

    # 将近80%的数据作为训练集，正向和负向各选取80%的数据作为训练集
    train_data = [(sent, 1) for sent in pos_sents[:4250]] + [(sent, 0) for sent in neg_sents[:4250]]
    # 约10%的数据作为验证集
    val_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
    # 约10%的数据作为测试集
    test_data = [(sent, 1) for sent in pos_sents[4800:]] + [(sent, 0) for sent in neg_sents[4800:]]


    # 随机洗牌，打乱顺序
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)


    print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

    # [s ...]为所有评论句子组成的list
    word_to_idx = utils.build_token_to_idx([s for s, _ in train_data + val_data + test_data])
    np.save('word2idx', word_to_idx)



class MR(Data.Dataset):
    """
    """
    pos_file = 'rt-polarity.pos'
    neg_file = "rt-polarity.neg"
    
    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        
        # already tokenized and there is no standard split
        # the size follow the Mou et al. 2016 instead
        pos_file = os.path.join(self.root, self.pos_file)
        neg_file = os.path.join(self.root, self.neg_file)

        # codecs.open(): 读入数据时，直接解码操作。防止编码格式问题。
        # .read(), .readlines(), .readline() 这些方法区别是：
        # read() 读取整个文件成一个字符串
        # readlines() 读取整个文件，自动将文件分成行的列表，供for line in fp.readlines() 调用
        # readline() 读取一行数据，速度慢。通常在内存不够的情况下使用
        pos_sents = codecs.open(pos_file, 'r', 'utf8').read().split('\n')
        # print(pos_sents[0:2])
        neg_sents = codecs.open(neg_file, 'r', 'utf8').read().split('\n')

        random.seed(1000)
        random.shuffle(pos_sents)
        random.shuffle(neg_sents)

        # print(len(pos_sents))
        ''' 5331 '''
        # print(len(neg_sents))
        ''' 5331 '''

        # 将近80%的数据作为训练集，正向和负向各选取80%的数据作为训练集
        train_data = [(sent, 1) for sent in pos_sents[:4250]] + [(sent, 0) for sent in neg_sents[:4250]]
        # 约10%的数据作为验证集
        val_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
        # 约10%的数据作为测试集
        test_data = [(sent, 1) for sent in pos_sents[4800:]] + [(sent, 0) for sent in neg_sents[4800:]]

        # 随机洗牌，打乱顺序
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)


        # print('train:', len(train_data), 'val:', len(val_data), 'test:', len(test_data))

        if train:
            self.items = train_data
        else:
            self.items = test_data
        # return train_data, val_data, test_data, word_to_idx, classes_to_idx

    def __getitem__(self, index):
        sent, label = self.items[index]
        return sent, label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':

    build_word_to_idx()



