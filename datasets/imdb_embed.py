# encoding: utf-8
""" 自定义数据集
我们要使用的训练集是imdb电影评论数据集。
这个集合有25,000个电影评论，有12,500个正面评论和12,500个负面评论。
"""

import torch
import torch.utils.data as Data
import torchvision

import os
import numpy as np
import codecs
import random

from . import utils
# import utils

random.seed(1000)

"""
# 评论句子对应的词向量索引矩阵，需要用到 wordsList.npy
words_list = np.load('datasets/wordsList.npy')  # TODO
print('Loaded the word list')
words_list = words_list.tolist() # Originally loaded as numpy array
words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8
'''
['0' ',' '.' ..., 'rolonda' 'zsombor' 'unk']
'''
"""
import torch.nn as nn
from torch.autograd import Variable

def prepare_embed():
    root = '../data/imdb'
    root = os.path.expanduser(root)

    classes, class_to_idx = utils.find_classes(root)

    ids_matrix = np.load('../data/idsMatrix.npy').tolist()
    print(len(ids_matrix))
    ''' 25000 '''
    
    word_vectors = np.load('../data/wordVectors.npy')
    embedding = nn.Embedding(400000, 50)
    embedding.weight.data = torch.from_numpy(word_vectors)
    input = Variable(torch.LongTensor(ids_matrix))
    print(input.size())
    embed_matrix = embedding(input)
    print('embed')
    data = embed_matrix.data.numpy()
    labels = []
    for i, _ in enumerate(data):
        if i <= 12500-1:
            labels.append(1)
        else:
            labels.append(0)
    
    
    dataset = list(zip(data, labels))
    #shuffle works inplace and returns None . 
    random.shuffle(dataset)
    print('shuffle')


    total = len(dataset)
    training = dataset[:int(total*0.8)]
    test = dataset[int(total*0.9):]

    training = list(zip(*training))
    test = list(zip(*test))

    np.save('train_data', training)
    print('save train_data success')
    np.save('test_data', test)
    print('save test_data success')


class IMDB_EMBED(Data.Dataset):
    def __init__(self, root, transform=None, train=True, max_seq_length=250):
        self.root = os.path.expanduser(root)

        classes, class_to_idx = utils.find_classes(root)                

        self.transform = transform
        self.train = train
        self.max_seq_length = max_seq_length
        self.classes = classes
        self.class_to_idx = class_to_idx


        if self.train:  # TODO 这里存在bug, 后面的都为 positiveReviews
            self.train_data, self.train_labels = np.load('datasets/train_data.npy')
        else:
            self.test_data, self.test_labels = np.load('datasets/test_data.npy')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            embed_sentence, label = self.train_data[index], self.train_labels[index]
        else:
            embed_sentence, label = self.test_data[index], self.test_labels[index]
        return embed_sentence, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


if __name__ == '__main__':
    # prepare_embed()
    
    train_data, train_labels = np.load('train_data.npy')
    print(train_labels)
    exit()

    dataset = IMDB_EMBED(root='../data/imdb', train=True, max_seq_length=250)
    print(dataset[0])
    exit()

    data, label = dataset[0]
    print(data)

    word_vectors = np.load('wordVectors.npy')
    ''' (400000, 50) '''
    '''
    [[ 0.          0.          0.         ...,  0.          0.          0.        ]
     ...,
     [-0.79149002  0.86616999  0.11998    ..., -0.29995999 -0.0063003
       0.39539999]]
    '''
    import torch.nn as nn
    from torch.autograd import Variable

    embedding = nn.Embedding(400000, 50)
    embedding.weight.data = torch.from_numpy(word_vectors)

    input = Variable(torch.LongTensor(data))
    embed = embedding(input)
    print(embed)
    exit()

