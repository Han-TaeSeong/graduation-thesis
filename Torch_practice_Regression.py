# torch.autograd 자동 미분 함수
# torch.nn 데이터 구조, 레이어 등
# torch.optim 최적화 알고리즘 ex)확률적 경사 하강법(stochastic gradient descent)
# torch.utils.data
# torch.onnx
# https://wikidocs.net/57805 참조

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Numpy로 텐서 만들기
# t = np.array([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]])
# print(t)
# print(t.ndim)
# print(t.shape)
# print('t[0] t[1] t[-1] = ', t[0], t[1], t[-2])
# print(t[:-1])


# pytorch로 텐서 만들기
# x = torch.randn(2, 3)

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
# nb_epochs = 1
#
# for epoch in range(nb_epochs+1):
#     hypothesis = x_train * w + b
#     cost = torch.mean((hypothesis - y_train) ** 2)
#
#     optimizer.zero_grad()    ## 미분전
#     print(w.grad)
#     cost.backward()         ## 미분
#     print(w.grad)
#     optimizer.step()        ## Gradient Descent을 모델에에적용
#     print(w.item())
#
#     if epoch%100 == 0:
#         print(f'Epoch : {epoch:4d}/{nb_epochs} W : {w.item():.3f} b : {b.item():.3f} cost : {cost.item():.6f}')


########### 자동미분
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


################# nn.Module로 선형회귀 구현
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
#
# model = nn.Linear(1, 1)
# print(list(model.parameters()))  # parameter보기
#
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# nb_epochs = 2000
#
# for epoch in range(nb_epochs + 1):
#     prediction = model(x_train)
#     cost = F.mse_loss(prediction, y_train)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch%100 ==0:
#         print(f'Epoch {epoch:4d}/{nb_epochs}  Cost : {cost.item():.6f}')
#
# new_var = torch.FloatTensor([4])
# pred_y = model(new_var)
#
# print(pred_y)
# print(list(model.parameters()))


####### Module로 다중 선형회귀 구현
# torch.manual_seed(1)
# x_train = torch.FloatTensor([[73, 80, 75],
#                              [93, 88, 93],
#                              [89, 91, 90],
#                              [96, 98, 100],
#                              [73, 66, 70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#
# model = nn.Linear(3, 1)
# print(list(model.parameters()))
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
# nb_epoch = 2000
#
# for epoch in range(nb_epoch + 1):
#
#     prediction = model(x_train)
#     cost = F.mse_loss(prediction, y_train)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(f'Epoch  {epoch:4d}/{nb_epoch}   Cost : {cost.item():.6f}')
#
# new_var = torch.FloatTensor([73,80,75])
# pred_y = model(new_var)
# print(f'########  {pred_y}')
# print(f'###### {list(model.parameters())}')


######## 모델을 클래스로 구현하기
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
#
#
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
#
# model = LinearRegressionModel()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# nb_epochs = 2000
# for epoch in range(nb_epochs + 1):
#     prediction = model(x_train)
#     cost = F.mse_loss(prediction, y_train)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
# new_var = torch.FloatTensor([4])
# print(model.forward(new_var).item())


###### Batch  // TensorDataset , Dataloader  이용
# 배치 크기는 보통 2의 제곱수를 사용합니다. ex) 2, 4, 8, 16, 32, 64...
# 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.
# x_train = torch.FloatTensor([[73, 80, 75],
#                              [93, 88, 93],
#                              [89, 91, 90],
#                              [96, 98, 100],
#                              [73, 66, 70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#
# dataset = TensorDataset(x_train, y_train)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# model = nn.Linear(3, 1)
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
# nb_epochs = 20
#
# for epoch in range(nb_epochs + 1):
#     for batch_index, samples in enumerate(dataloader):
#         print(f'batch index : {batch_index}   samples : {samples}')
#         x_train, y_train = samples
#         prediction = model(x_train)
#         cost = F.mse_loss(prediction, y_train)
#
#         optimizer.zero_grad()
#         cost.backward()
#         optimizer.step()
#
#         print(f'Epoch : {epoch:4d}/{nb_epochs}  Batch : {batch_index + 1}/{len(dataloader)}  Cost : {cost.item()}')


####### Custom Dataset 커스텀 데이터셋
# class CustomDataset(Dataset):
#     def __init__(self):
#         self.x_data = [[73, 80, 75],
#                        [93, 88, 93],
#                        [89, 91, 90],
#                        [96, 98, 100],
#                        [73, 66, 70]]
#         self.y_data = [[152], [185], [180], [196], [142]]
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         x = torch.FloatTensor(self.x_data[idx])
#         y = torch.FloatTensor(self.y_data[idx])
#         return x, y
#
# dataset = CustomDataset()
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# print(dataset.__getitem__(1))
# print(dataset.__len__())
#
# model = nn.Linear(3,1)
# optimizer = optim.SGD(model.parameters(),lr=1e-5)
# nb_epochs = 20
#
# for epoch in range(nb_epochs+1):
#     for batch_index, samples in enumerate(dataloader):
#
#         x_train, y_train = samples
#         prediction = model(x_train)
#         cost = F.mse_loss(prediction,y_train)
#
#         optimizer.zero_grad()
#         cost.backward()
#         optimizer.step()
#
# new_var = torch.FloatTensor([73,80,75])
# pred_y = model(new_var)
# print(f'예측값 : {pred_y.item()}')


########## 시그모이드 함수 그리기    ##### 로지스틱 회귀
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# x = np.arange(-5, 5, 0.1)
# y = sigmoid(x)

# plt.plot(x,y,'r')
# plt.plot([0,0],[0,1],':')
# plt.title('Sigmoid Function')
# plt.show()

# y1 = sigmoid(0.5 * x)
# y2 = sigmoid(x)
# y3 = sigmoid(2 * x)
#
# plt.plot(x, y1, 'r', linestyle='--')
# plt.plot(x, y2, 'g')
# plt.plot(x, y3, 'b', linestyle='--')
# plt.plot([0,0],[0,1],':')
# plt.title('Sigmoid Function')
# plt.show()

# z1 = sigmoid(x)
# z2 = sigmoid(x+1)
# plt.plot(x, z1, 'r', linestyle='--')
# plt.plot(x, z2, 'g')
# plt.plot([0,0],[0,1],':')
# plt.title('Sigmoid Function')
# plt.show()


########### 로지스틱 회귀 구현
# torch.manual_seed(1)
# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
# x_train = torch.FloatTensor(x_data)
# y_train = torch.FloatTensor(y_data)
#
# w = torch.zeros((2,1),requires_grad=True)
# b = torch.zeros(1,requires_grad=True)
#
# optimizer = optim.SGD([w,b],lr=1)
# nb_epochs = 1000
#
# for epoch in range(nb_epochs+1):
#     hypothesis = 1/(1+torch.exp(-(x_train.matmul(w)+b)))
#     cost = -(y_train*torch.log(hypothesis)+(1-y_train)*(torch.log(1-hypothesis))).mean()
#     ################# Binary Cross Entropy#########################
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch%100==0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
#
# hypothesis = torch.sigmoid(x_train.matmul(w)+b)
# prediction = hypothesis >= torch.FloatTensor([0.5])  ############### False , True 쉽게
# print(prediction)


############ nn.Module을 이용한 로지스틱 회귀분석
# torch.manual_seed(1)
# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
# x_train = torch.FloatTensor(x_data)
# y_train = torch.FloatTensor(y_data)
#
# model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
# optimizer = optim.SGD(model.parameters(),lr=1)
# nb_epochs = 1000
#
# for epoch in range(nb_epochs+1):
#     prediction = model(x_train)
#     cost = F.binary_cross_entropy(prediction,y_train)
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if epoch%100==0:
#         pred = prediction >= torch.FloatTensor([0.5])
#         correct_pred = y_train == pred
#         accuracy = correct_pred.sum().item()/len(correct_pred)
#         print(f'accuracy : {accuracy*100:.2f}%')
# print(model(x_train))
# print(list(model.parameters()))


############# 로지스틱 클래스로 구현
torch.manual_seed(1)
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)
np_epochs = 1000

for epoch in range(np_epochs + 1):
    prediction = model(x_train)
    cost = F.binary_cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        pred = prediction >= torch.FloatTensor([0.5])
        correct_pred = y_train == pred
        accuracy = correct_pred.sum().item() / len(correct_pred)
        print(f'accuracy : {accuracy * 100:.2f}%')




############# 소프트맥스 회귀 시작  원-핫 인코딩부터  1,2,3,4로한다면 cost를 구할때 상관성이 첨가가 된다.
#Softmax에서 e를 쓰는 이유
# 자연상수 e를 쓰는 이유는 두 가지
# 1. 미분이 용이 -> gradient를 발생시켜 역전파~
# 2. 입력 벡터가 더 잘 구분되게(큰 값은 더 크게, 작은 값은 더 작게) //지수함수니까
# torch.manual_seed(3)
# a = torch.FloatTensor([1, 2, 3])
# hypothesis = F.softmax(a,dim=0)
# print(hypothesis)
# print(hypothesis.sum())   ##합은 =1

# z = torch.rand((3,5),requires_grad=True)
# hypothesis = F.softmax(z, dim=1)
# y = torch.randint(5,(3,)).long()
# y_one_hot = torch.zeros_like(hypothesis)
# y_one_hot.scatter_(1, y.unsqueeze(1), 1)