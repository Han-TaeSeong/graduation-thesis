import numpy as np

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx as onnx

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# device = torch.device("cuda")
#
# model = torch.load('linear.pth')
#
# mnist_train = dsets.MNIST(root='MNIST_data/',
#                           train=True,
#                           transform=transforms.ToTensor(),
#                           download=True)
#
# mnist_test = dsets.MNIST(root='MNIST_data/',
#                          train=False,
#                          transform=transforms.ToTensor(),
#                          download=True)
#
# x_test = mnist_test.data[9].view(-1, 28 * 28).float().to(device)
# y_test = mnist_test.targets[9].to(device)
#
# print(y_test)
# model.train()
# print(torch.argmax(model(x_test)))



#
# x = torch.tensor(digits.data,dtype=torch.float32)
# y = torch.tensor(digits.target,dtype=torch.int64)

# a = torch.Tensor([[1,2],[3,4],[5,6]])
# b = torch.Tensor([1,2,3,4,5,6])
# print(b.view_as(a))


class A(object):
    def do_work(self):
        print('A의 do_work')


class B(A):  # A를 상속받는 클래스
    def do_work(self, x):
        print(1)
        super(B, self).do_work()
        print(2)
        super(A, self).do_work()