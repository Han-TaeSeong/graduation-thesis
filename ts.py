import numpy as np
import pandas as pd
import math
import pytorch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('data.xlsx',usecols="A:E") 

df.loc[df['age']>150,['age']]=np.nan
df.loc[df['target']=='unknown',['target']]='class4'


y_arr = df['target'].values
num_y=0
dic_y={}
for ith_y in sorted(y_arr):
    dic_y[ith_y]=num_y
    num_y +=1


df['target']=df['target'].replace(dic_y)
df=df.fillna(0)
df.loc[df['gender']=='unknown',['gender']]='U'
df.loc[df['hobby']=='unknown',['hobby']]='U'
df.loc[df['month_birth']==21,['month_birth']]=0
######################

std = StandardScaler()
x_std2 = std.fit_transform(df[['month_birth']])
print(x_std2)
print(f' {math.trunc(np.mean(x_std2))} \n {np.std(x_std2)}')

