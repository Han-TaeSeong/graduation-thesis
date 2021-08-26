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

device = torch.device("cuda")
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)


########### 단층 퍼셉트론
# def OR_gate(x1, x2):
#     w1 = 0.6
#     w2 = 0.6
#     b = -0.5
#     result = x1 * w1 + x2 * w2 + b
#     if result <= 0:
#         return 0
#     else:
#         return 1
#
#
# def NAND_gate(x1, x2):
#     w1 = -0.5
#     w2 = -0.5
#     b = 0.7
#     result = x1 * w1 + x2 * w2 + b
#     if result <= 0:
#         return 0
#     else:
#         return 1
#
#
# def AND_gate(x1, x2):
#     w1 = 0.5
#     w2 = 0.5
#     b = -0.7
#     result = x1 * w1 + x2 * w2 + b
#     if result <= 0:
#         return 0
#     else:
#         return 1
#
#
# ########## 다층퍼셉트론이 필요한 이유 : 단층으로는  XOR 구현 불가
# def XOR_gate(x1, x2):
#     X = OR_gate(x1, x2)
#     Y = NAND_gate(x1, x2)
#     result = AND_gate(X, Y)
#     if result <= 0:
#         return 0
#     else:
#         return 1
#
#
# print(XOR_gate(0, 0), XOR_gate(0, 1), XOR_gate(1, 0), XOR_gate(1, 1))

############ 단층 퍼셉트론 파이토치로 구현(xor) 해를 구할수없다
# X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
# Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
#
# linear = nn.Linear(2, 1)
# sigmoid = nn.Sigmoid()
# model = nn.Sequential(linear, sigmoid).to(device)
#
# criterion = nn.BCELoss()
#
# optimizer = optim.SGD(model.parameters(), lr=1)
#
# for epoch in range(1001):
#     hypothesis = model(X)
#     cost = criterion(hypothesis, Y)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(epoch, cost.item())
#
# with torch.no_grad():
#     hypothesis = model(X)
#     pred = (hypothesis>0.5).float()
#     accuracy = (pred == Y).float().mean()
#     print(hypothesis.detach().cpu().numpy())
#     print(accuracy)

####### 다층 퍼셉트론 파이토치로 구현(xor)
