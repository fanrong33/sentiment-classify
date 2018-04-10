# encoding: utf-8
""" 自定义数据集 HOTEL
@version 1.0.0 build 20180409
"""

import torch
import torch.utils.data as Data
import torchvision

import os
import numpy as np
import codecs
import random

from . import utils



class HOTEL(Data.Dataset):
    training_file = "training_data.npy"
    test_file = "test_data.npy"

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)

        self.train = train

        if self.train:
            self.train_data, self.train_labels = np.load(os.path.join(root, self.training_file))
        else:
            self.test_data, self.test_labels = np.load(os.path.join(root, self.test_file))

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

    train_dataset = HOTEL(root='../data/hotel/', train=True)
    print(len(train_dataset))

    (data, label) = train_dataset[0]
    print(data)
    print(label)



