# torch.autograd 자동 미분 함수
# torch.nn 데이터 구조, 레이어 등
# torch.optim 최적화 알고리즘 ex)확률적 경사 하강법(stochastic gradient descent)
# torch.utils.data
# torch.onnx

import numpy as np

# Numpy로 텐서 만들기
# t = np.array([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
# print(t)
# print(t.ndim)
# print(t.shape)
# print('t[0] t[1] t[-1] = ', t[0], t[1], t[-2])
# print(t[:-1])


# pytorch로 텐서 만들기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1차원
# t = torch.FloatTensor([0,1,2,3,4,5,6])
# print(t)
# print(t.dim())
# print(t.size())


# 2차원
# t = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print(t)
# print(t.dim())
# print(t.size())
# print(t[:,1])
# print(t[:,1].size())


# 행렬곱 matmul 원소곱mul
# m1 = torch.FloatTensor([[1, 2], [3, 4]])
# m2 = torch.FloatTensor([[1], [2]])
# print(m1*m2)
# print(m1.matmul(m2))


# 평균//원소덧셈
# m1 = torch.FloatTensor([[1, 2], [3, 4]])
# print(m1)
# print(m1.mean())
# print(m1.mean(dim=0)) #행의평균 #행을1로
# print(m1.sum())
# print(m1.sum(dim=1)) #열의평균 #열을1로


# Max, ArgMax
# m1 = torch.FloatTensor([[1, 2], [9, 4]])
# print(m1.max(dim=0))  #행중 최대 #행을1로 #[1,1]의 의미 : 0열:index 1 1열:index 2
# m2 = torch.FloatTensor([[1, 8], [3, 4]])
# print(m2.max(dim=0)[1])
# print(m1.argmax())


# 텐서의 크기변경
# t = np.array([[[0, 1, 2],
#                [3, 4, 5]],
#               [[6, 7, 8],
#                [9, 10, 11]],
#               [[12,13,14],
#               [15,16,17]]])
# ft = torch.tensor(t)
# print(ft.shape)
# print(ft.view(-1,3)) ## 행의 크기는 알아서 정해라
# print(ft.view(-1,3).shape)
# print(ft.unsqueeze(1))
# print(ft.unsqueeze(1).shape)


# # TypeCasting
# lt = torch.LongTensor([1,2,3,4])
# print(lt.float())
# bt = torch.ByteTensor([True, False, 11, True])
# print(bt)


# Concatenate
# x = torch.FloatTensor([[1, 2], [3, 4]])
# y = torch.FloatTensor([[5, 6], [7, 8]])
# print(torch.cat([x,y],dim=1))
# x = torch.FloatTensor([1, 4])
# y = torch.FloatTensor([2, 5])
# z = torch.FloatTensor([3, 6])
# print(torch.stack([x,y,z],dim=1))


# 0, 1 Tensor
# x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
# print(torch.ones_like(x))
# x = torch.FloatTensor([[1, 2], [3, 4]])
# y = torch.tensor([2,3]).view(2,1).float()
# print(x.matmul(y))
# print(x)


# class 연습
# result = 0
# def add(num):
#     global result
#     result += num
#     return result
# print(add(3))
# print(add(10))


# class Calculator:
#     def __init__(self):
#         self.result = 0
#
#     def add(self, num):
#         self.result += num
#         return self.result
# a= Calculator()
# print(a.add(50))
# print(a.add(11))


######################### 선형회귀
# torch.manual_seed(1)
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
# w = torch.zeros(1, requires_grad=True)  # w.item()
# b = torch.zeros(1, requires_grad=True)
#
# optimizer = optim.SGD([w, b], lr=0.01)
# nb_epochs = 2000
#
# for epoch in range(nb_epochs+1):
#     hypothesis = x_train * w + b
#     cost = torch.mean((hypothesis - y_train) ** 2)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch%100 == 0:
#         print(f'Epoch : {epoch:4d}/{nb_epochs} W : {w.item():.3f} b : {b.item():.3f} cost : {cost.item():.6f}')


# 자동미분
# w = torch.tensor(10.0,requires_grad=True)
# y = w**2
# z = 2*y+5
# z.backward()
# print(f'dz/dw : {w.grad}')


############## 다중선형회귀
# torch.manual_seed(1)
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#
# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
# nb_epochs = 1000
#
# for epoch in range(nb_epochs+1):
#     hypothesis = w1*x1_train + w2*x2_train + w3*x3_train +b
#     cost = torch.mean((hypothesis-y_train)**2)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(f'Epoch : {epoch:4d}/{nb_epochs} W1 : {w1.item():.3f} W2 : {w2.item():.3f} W3 : {w3.item():.3f} '
#               f'b : {b.item():.3f} cost : {cost.item():.6f}')




############# 다중선형회귀 Matrix로 구현
# x_train = torch.FloatTensor([[73, 80, 75],
#                              [93, 88, 93],
#                              [89, 91, 80],
#                              [96, 98, 100],
#                              [73, 66, 70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# w = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# optimizer = optim.SGD([w, b], lr=1e-5)
# nb_epochs = 1000
#
# for epoch in range(nb_epochs+1):
#     hypothesis = x_train.matmul(w)+b
#     cost = torch.mean((hypothesis-y_train)**2)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(f'Epoch : {epoch:4d}/{nb_epochs} W1 : {w.squeeze().detach()} Hypo : {hypothesis.squeeze().detach()}'
#               f'b : {b.item():.3f} cost : {cost.item():.6f}')

