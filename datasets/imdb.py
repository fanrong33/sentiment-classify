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

import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


class IMDB(Data.Dataset):
    def __init__(self, root, transform=None, train=True, max_seq_length=250):
        self.root = os.path.expanduser(root)

        classes, class_to_idx = utils.find_classes(root)

        train_data = []
        test_data = []
        for target in sorted(os.listdir(self.root)):
            items = []
            d = os.path.join(self.root, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    # if is_image_file(fname):
                    path = os.path.join(root, fname)
                    sentence = codecs.open(path, 'r', 'utf8').read() # 以空格分割的原始句子列表, 兼容中英文
                    item = (sentence, class_to_idx[target])  # tuple()
                    items.append(item)

            total = len(items)
            train_num = int(total*0.8)
            train_data = train_data + items[:train_num]
            test_data = test_data + items[int(total*0.9):]

        random.shuffle(train_data)
        random.shuffle(test_data)
        
        
        self.transform = transform
        self.train = train
        self.max_seq_length = max_seq_length
        self.classes = classes
        self.class_to_idx = class_to_idx

        if self.train:  # TODO 这里存在bug, 后面的都为 positiveReviews
            self.items = train_data
        else:
            self.items = test_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sentence, label = self.items[index]
        
        return sentence, label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    dataset = IMDB(root='../data/imdb', max_seq_length=250)
    print(dataset[0])
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

