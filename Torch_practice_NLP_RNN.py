import math
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx as onnx

from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models

from torchtext.legacy import data, datasets
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator  ##batch만큼 데이터 로드
from torchtext.legacy.vocab import Vectors  ##word2vec 사전훈련 데이터
from torchtext.vocab import GloVe  ## Glove 사전훈련 데이터

from konlpy.tag import Mecab  ## 한국어 자연어 처리
import spacy  ## 영어 자연어 처리
import nltk  ## 영어 자연어 처리
from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize  ###sentence, word tokenize

import zipfile
from lxml import etree
import re
import urllib.request
from gensim.models import Word2Vec, KeyedVectors
import gensim

# from glove import Corpus, Glove
####pip install glove_python_binary

# ###### 영어 토큰화 띄어쓰기만 하면 된다.
# en_text = 'A Dog Run back corner near spare bedrooms'
# print(en_text.split())
# print(list(en_text))
# ##### 한글 토큰화 조사 때문에 띄어쓰기만 하면 단어 집합이 커진다
# kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 " \
#            "사과랑 오렌지를 사왔어"
# print(kor_text.split())
#
# tokenizer = Mecab(dicpath='C:\\Mecab\\mecab-ko-dic')
# print(tokenizer.morphs(kor_text))


# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt",filename='ratings.txt')
# data = pd.read_table('ratings.txt')
# # print(data.head())
# # print(len(data))
# # print(data.shape)
# sample_data = data[:100].copy()      ##### DataFrame에 대한 이해필요
# sample_data.loc[:,'document'] = sample_data.loc[:,'document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)   ## 정규식 Regex
#
# # print(sample_data.head())
#
# stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# tokenizer = Mecab('C:\\Mecab\\mecab-ko-dic')
#
# tokenized =[]
# for sentence in sample_data['document']:
#     temp = tokenizer.morphs(sentence)
#     temp = [word for word in temp if not word in stopwords]
#     tokenized.append(temp)
#
# # print(tokenized[:10])
# # print("#####################")
# # print(np.hstack(tokenized))
#
# vocab = FreqDist(np.hstack(tokenized))
# # print(len(vocab))
# vocab = vocab.most_common(500)  ## output: tuple (word,cnt)
# # print(vocab)
#
#
# ############  각 단어에 고유 정수 부여
# word_to_index = {word[0] : index+2 for index, word in enumerate(vocab)}  ### 딕셔너리 생성
# word_to_index['pad'] = 1
# word_to_index['unk'] = 0
# # print(word_to_index)
# encoded = []
# for line in tokenized:
#     temp = []
#     for w in line:
#         try:
#             temp.append(word_to_index[w])
#         except KeyError:
#             temp.append(word_to_index['unk'])
#     encoded.append(temp)
#
# # print(encoded[:10])    ##숫자
#
#
# ############## 문장들을 동일한 길이로 바꿔주는 패딩
# max_len = max(len(l) for l in encoded)
# print(f'리뷰의 최대길이 : {max_len}')
# print(f'리뷰의 최소길이 : {min(len(l) for l in encoded)}')
# print(f'리뷰의 평균길이 : {sum(map(len,encoded))/len(encoded)}')
#
# # plt.hist([len(s) for s in encoded],bins=50)
# # plt.xlabel('length of sample')
# # plt.ylabel('number of sample')
# # plt.show()
#
# for line in encoded:
#     if len(line) < max_len:
#         line += [word_to_index['pad']] * (max_len-len(line))
# print(encoded[:3])


###################### TorchText  영어
### tokenize -> vocab(단어집합) -> 정수 인코딩
# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv"
#                            ,filename="IMDB_Reviews.csv")
# df = pd.read_csv('IMDB_Reviews.csv',encoding='latin1')
# print(df.head())
# # train_df = df[:25000].copy()
# # test_df = df[25000:].copy()
# # train_df.to_csv("train_data.csv",index=False)
# # test_df.to_csv('test_data.csv',index=False)
#
# TEXT = data.Field(sequential=True, use_vocab= True, tokenize=str.split,lower =True,
#                   batch_first=True,fix_length=20)
# LABEL = data.Field(sequential=False,use_vocab=False,batch_first=False,is_target=True)
#
# train_data, test_data = TabularDataset.splits(train='train_data.csv',test='test_data.csv',path='.',format='csv',
#                                              fields=[('text',TEXT),('label',LABEL)],skip_header=True)
#
# print(vars(train_data[0]))
# # print(train_data.fields.items())
#
# TEXT.build_vocab(train_data,min_freq=10,max_size=10000)
# print(TEXT.vocab.stoi)
#
# batch_size=5
# train_loader = Iterator(dataset=train_data,batch_size=batch_size,shuffle=False)
# test_loader = Iterator(dataset=test_data,batch_size=batch_size)
#
# batch = next(iter(train_loader))
# # print(type(batch))
# print(batch.text[0])


############### TorchText 한국어
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
#
# train_df = pd.read_table('ratings_train.txt')
# test_df = pd.read_table('ratings_test.txt')
#
# tokenizer = Mecab(dicpath='C:\\Mecab\\mecab-ko-dic')
#
# ID = data.Field(sequential=False, use_vocab=False)
# TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer.morphs, lower=True, batch_first=True,fix_length=20)
# LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)
#
# train_data, test_data = TabularDataset.splits(path='.',train='ratings_train.txt',test='ratings_test.txt',format='tsv',
#                                               fields=[('id',ID),('text',TEXT),('labet',LABEL)],skip_header=True)
#
# print(vars(train_data[0]))
#
# TEXT.build_vocab(train_data,min_freq=10,max_size=10000)
# print(TEXT.vocab.stoi)
#
# batch_size = 5
# train_loader = Iterator(dataset=train_data,batch_size=batch_size)
# test_loader = Iterator(dataset=test_data,batch_size=batch_size)

# batch = next(iter(train_loader))
# print(batch.text)


########### 자연어 처리에서의 원 핫 인코딩
# tokenizer = Mecab(dicpath='C:\\Mecab\\mecab-ko-dic')
# token = tokenizer.morphs('나는 자연어 처리를 배운다')
#
# word2index = {}
# for voca in token:
#     if voca not in word2index.keys():
#         word2index[voca] = len(word2index)
# print(word2index)
#
# def one_hot_vector(word,word2index):
#     one_hot_vector = [0]*len(word2index)
#     index = word2index[word]
#     one_hot_vector[index] = 1
#     return one_hot_vector
# print(one_hot_vector('자연어',word2index))


############## 워드 임베딩 word embedding  영어  word2vec
# dog = torch.FloatTensor([1, 0, 0, 0, 0])
# cat = torch.FloatTensor([0, 1, 0, 0, 0])
# computer = torch.FloatTensor([0, 0, 1, 0, 0])
# netbook = torch.FloatTensor([0, 0, 0, 1, 0])
# book = torch.FloatTensor([0, 0, 0, 0, 1])
#
# print(torch.cosine_similarity(dog, cat, dim=0))
# print(torch.cosine_similarity(cat, computer, dim=0))
# print(torch.cosine_similarity(computer, netbook, dim=0))
# print(torch.cosine_similarity(netbook, book, dim=0))


# urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/"
#                            "master/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# targetXML = open('ted_en-20160408.xml','r',encoding='UTF-8')
# target_text = etree.parse(targetXML)
# parse_text = '\n'.join(target_text.xpath('//content/text()'))
#
# content_text = re.sub(r'\([^)]*\)', '', parse_text)   ##Raw String  규칙
# sent_text = sent_tokenize(content_text)
#
# normalized_text = []
# for string in sent_text:
#     tokens = re.sub(r"[^a-z0-9]+"," ",string.lower())
#     normalized_text.append(tokens)
#
# result = []
# result = [word_tokenize(sentence) for sentence in normalized_text]
# print(len(result))
# for line in result[:3]:
#     print(line)
#
# model = Word2Vec(sentences=result,vector_size=100,window=5,min_count=5,workers=4,sg=0)
# model_result = model.wv.most_similar("man")
# print(model_result)
#
# #######################모델 저장하기
# model.wv.save_word2vec_format('./eng_w2v')
# loaded_model = KeyedVectors.load_word2vec_format('eng_w2v')
#
# model_result = loaded_model.most_similar("man")
# print(model_result)
#
#
# ################ 사전 임베딩 데이터 구글에서 가져오기
# model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)
# print(model.vectors.shape)
# print(model.similarity('this','is'))
# print(model.most_similar('cat'))
#
# model = gensim.models.Word2Vec.load('.\\ko\\ko.bin') #### pip install gensim==3.8.1
# print(model.wv.most_similar("강아지"))
# print(model.wv.similarity("강아지","야구"))


###################  임베딩 벡터의 시각화
# python -m gensim.scripts.word2vec2tensor --input eng_w2v --output eng_w2v
# https://projector.tensorflow.org/


######################### nn.Embedding 사용 안하고 구현
# train_data = 'you need to know how to code'
# word_set = set(train_data.split())
# vocab = {word : i+2 for i, word in enumerate(word_set)}
# vocab['<unk>']=0
# vocab['<pad>']=1
# print(vocab)
#
# embedding_table = torch.rand(8,3)
# print(embedding_table)
#
# sample = 'you need to run'.split()
#
# indexs = []
# for word in sample:
#     try:
#         indexs.append(vocab[word])
#     except KeyError:
#         indexs.append(vocab['<unk>'])
#
# indexs = torch.LongTensor(indexs)
# print(indexs)
#
# lookup_result = embedding_table[indexs,:]
# print(lookup_result)

########### nn.Embedding 사용
# train_data = 'you need to know how to code'
# word_set = set(train_data.split())
# vocab = {word : i+2 for i, word in enumerate(word_set)}
# vocab['<unk>']=0
# vocab['<pad>']=1
#
# embedding_layer = nn.Embedding(num_embeddings=len(vocab),embedding_dim=3,padding_idx=1)
# print(embedding_layer.weight)


############# torch 사전 임베딩 사용
# TEXT = data.Field(sequential=True, batch_first=True, lower=True)
# LABEL = data.Field(sequential=False,batch_first=True)
#
# trainset, testset = datasets.IMDB.splits(TEXT,LABEL)
# print(len(trainset))
#
# word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')
# vectors = Vectors(name='eng_w2v')
# print(word2vec_model['this'])
#
# TEXT.build_vocab(trainset,vectors=vectors,max_size=10000,min_freq=10)
#
# print(TEXT.vocab.stoi)
# print(TEXT.vocab.vectors.shape)
#
# embedding_lyaer = nn.Embedding.from_pretrained(TEXT.vocab.vectors,freeze=false)


####################  Glove 사전 임베딩 사용
# TEXT = data.Field(sequential=True, batch_first=True, lower=True)
# LABEL = data.Field(sequential=False,batch_first=True)
#
# trainset, testset = datasets.IMDB.splits(TEXT,LABEL)
# TEXT.build_vocab(trainset,vectors=GloVe(name='6B',dim=300),max_size=10000,min_freq=10)
# LABEL.build_vocab(trainset)
#
# # print(TEXT.vocab.stoi)
# print(TEXT.vocab.vectors[10])
#
# embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors,freeze=False)


##########  RNN 로우레벨 구현
# timesteps = 10
# input_size = 4
# hidden_size = 8
#
# inputs = np.random.random((timesteps,input_size))
# hidden_state_t = np.zeros((hidden_size,))
#
# wx = np.random.random((input_size,hidden_size))
# wh = np.random.random((hidden_size,hidden_size))
# b = np.random.random((hidden_size,))
#
# total_hidden_states =[]
#
# for input_t in inputs:
#     output_t = np.tanh(np.matmul(input_t,wx)+np.matmul(hidden_state_t.transpose(),wh)+b)
#     total_hidden_states.append(list(output_t))
#     print(np.shape(total_hidden_states))
#     hidden_state_t = output_t
# print(f'total : {total_hidden_states}')
# total_hidden_states = np.stack(total_hidden_states,axis=0)
# print(f'stack : {total_hidden_states}')

############# nn.RNN 사용
# input_size = 5
# hidden_size = 8
#
# inputs = torch.Tensor(1,10,5)   ##batch X Time X Dimension
# cell = nn.RNN(input_size,hidden_size,batch_first=True)
# outputs , status = cell(inputs)
# print(outputs.shape)
# print(status.shape)

############## Deep RNN
# inputs = torch.Tensor(1,10,5)
# cell = nn.RNN(input_size=5,hidden_size=8,num_layers=2,batch_first=True)
# outputs, status = cell(inputs)
# print(outputs.shape)
# print(status.shape)


############### Bi-directional-RNN
# inputs = torch.Tensor(1,10,5)
# cell = nn.RNN(input_size=5,hidden_size=8,num_layers=2,batch_first=True,bidirectional=True)
# output, status = cell(inputs)
# print(output.shape)
# print(status.shape)


###########  Char RNN
# input_str = 'apple'
# a = 'aa bb cc ddf'
# label_str = 'pple!'
# char_vocab = sorted(list(set(input_str+label_str)))
# # print(char_vocab)
# vocab_size = len(char_vocab)
# input_size = len(char_vocab)
# hidden_size = 5
# output_size = 5
# learning_rate = 0.1
#
# char_to_index = dict((c,i) for i, c in enumerate(char_vocab))
# # print(char_to_index)
#
# index_to_char={}
# for key, value in char_to_index.items():
#     index_to_char[value]=key
# # print(index_to_char)
#
# x_data = [char_to_index[c] for c in input_str]
# y_data = [char_to_index[c] for c in label_str]
# x_data = [x_data]
# y_data = [y_data]
#
# x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
# # print(x_one_hot)
#
# x = torch.FloatTensor(x_one_hot)
# y = torch.LongTensor(y_data)
#
# # print(x.shape, y.shape)
#
#
# class Net(nn.Module):
#     def __init__(self,input_size,hidden_size,output_size):
#         super().__init__()
#         self.rnn = nn.RNN(input_size,hidden_size,batch_first=True)
#         self.fc = nn.Linear(hidden_size,output_size)
#
#     def forward(self,x):
#         x, status = self.rnn(x)
#         x = self.fc(x)
#         return x
#
# net = Net(input_size,hidden_size,output_size)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),learning_rate)
#
# for i in range(100):
#     output = net(x)
#     loss = loss_fn(output.view(-1,input_size),y.view(-1))
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     result = output.argmax(axis=2)
#     result_srt = ''.join([index_to_char[c] for c in result.squeeze().numpy()])
#     print(f'pred = {result_srt}')


############ char RNN large data
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}

dic_size = len(char_dic)

hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->>', y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

x_one_hot = [np.eye(dic_size)[x] for x in x_data]
x = torch.FloatTensor(x_one_hot)
y = torch.LongTensor(y_data)

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim,hidden_dim,num_layers=layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,hidden_dim)

    def forward(self, x):
        x, status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(dic_size, hidden_size, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),learning_rate)


for i in range(100):
    output = net(x)
    loss = loss_fn(output.view(-1,dic_size),y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = output.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(result):
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]

    print(predict_str)


