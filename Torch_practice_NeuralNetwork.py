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

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)
from sklearn.model_selection import train_test_split

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
# X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
# Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
#
# model = nn.Sequential(nn.Linear(2, 10), nn.Sigmoid(), nn.Linear(10, 10), nn.Sigmoid(),
#                       nn.Linear(10, 10), nn.Sigmoid(), nn.Linear(10, 1), nn.Sigmoid()).to(device)
#
# criterion = nn.BCELoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=1)
#
# for epoch in range(10001):
#     hypothesis = model(X)
#     cost = criterion(hypothesis, Y)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch % 500 == 0:
#         print(epoch, cost.item())
#
# with torch.no_grad():
#     hypothesis = model(X)
#     predicted = (hypothesis>0.5).float()
#     accuracy = (predicted == Y).float().mean()
#
#     print(f'Hypothesis : {hypothesis.detach().cpu().numpy()}')
#     print(f'Predicted : {predicted.detach().cpu().numpy()}')
#     print(f'Y : {Y}')
#     print(f'Accuracy : {accuracy}')


########## 비선형 활성화 함수 (Activation Function)
# x=torch.arange(-5,5,0.1)
# y=torch.tanh(x)
#
# plt.plot(x,y)
# plt.axvline(x=0,color='orange',linestyle=':')
# plt.axhline(y=0,color='orange',linestyle="--")
# plt.title('Tanh Fucntion')
# plt.show()

# def relu(x):
#     return np.maximum(0,x)
# x=torch.arange(-5,5,0.1)
# y=relu(x)
#
# plt.plot(x,y)
# plt.axvline(x=0,color='orange',linestyle=':')
# plt.title('Relu Function')
# plt.show()


# def leakyrelu(x):
#     return np.maximum(0.1 * x, x)
#
#
# x = torch.arange(-5, 5, 0.1)
# y = leakyrelu(x)
#
# plt.plot(x, y)
# plt.axvline(x=0, color='orange', linestyle=':')
# plt.title('Relu Function')
# plt.show()


# x = np.arange(-2.0, 2.0, 0.1)
# y = np.exp(x) / np.sum(np.exp(x))
#
# plt.plot(x, y)
# plt.title('Softmax Function')
# plt.show()


#############  숫자 필기 데이터
# from sklearn.datasets import load_digits
#
# digits = load_digits()
# # # print(digits.images[0])
# # # print(digits.target[0])
# # # print(len(digits.images))
# #
# #
# # images_and_labels = list(zip(digits.images, digits.target))
# # for index, (image,label) in enumerate(images_and_labels[:6]):
# #     plt.subplot(2, 5, index+1)
# #     plt.axis('off')
# #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
# #     plt.title('Sample: %i'%index)
# # plt.show()
#
#
# x = torch.tensor(digits.data,dtype=torch.float32,requires_grad=True)
# y = torch.tensor(digits.target,dtype=torch.int64)
#
# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(64,32)
#         self.linear2 = nn.Linear(32,16)
#         self.linear3 = nn.Linear(16,10)
#
#     def forward(self,x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         return self.linear3(x)
#
# model = Classifier()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())
#
# losses = []
#
# for epoch in range(101):
#     pred=model(x)
#     loss = loss_fn(pred, y)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 10 ==0:
#         print(f'Epoch : {epoch}  Cost : {loss.item():.6f}')
#
#     losses.append(loss.item())
# plt.plot(losses)
# plt.show()


########### MNIST 데이터 분류하기
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
#
# loader_train = DataLoader(mnist_train, batch_size=64, shuffle=True)
# loader_test = DataLoader(mnist_test, batch_size=64, shuffle=False)
#
#
# class DNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(784, 100)
#         self.linear2 = nn.Linear(100, 100)
#         self.linear3 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         return self.linear3(x)
#
#
# model = DNN()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
def train(epoch):
    model.train()

    for data, targets in loader_train:
        data/=255
        pred = model(data.view(-1,28*28))
        loss = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch{epoch} : 완료')

def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:
            data /= 255
            output = model(data.view(-1,28*28))

            pred = torch.argmax(output,dim=1)
            correct += pred.eq(targets.data.view_as(pred)).sum()
    data_num = len(loader_test.dataset)
    print(f'정확도 : {correct/data_num*100:.2f}')

for epoch in range(3):
    train(epoch)
test()

model.eval()
data = mnist_test.data[2018].float()
output = model(data.view(-1,28*28))
pred = torch.argmax(output)

plt.imshow(mnist_test.data[2018].view(28,28),cmap='gray')
plt.show()
print(f'예측 : {pred}')
print(f"정답 : {mnist_test.targets[2018]}")






