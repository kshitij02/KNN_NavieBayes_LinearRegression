#!/usr/bin/env python
# coding: utf-8

# # Part 1

# In[21]:


import numpy as num
import pandas as pd
import math
import pprint
from numpy.linalg import inv
from sklearn.metrics import r2_score
import sys


# In[2]:


def normalize(data,list_attribute):
#     for attr in list_attribute:
#         print attr
#         attr_std=data[attr].std()
#         attr_mean=data[attr].mean()
#         li=[]
#         for index, row in data.iterrows():
#             attr_value=float((row[attr]-attr_mean))/attr_std
#             li.append(attr_value)
#         data.drop(columns=attr)
#         data[attr]=li
    norm = (data - data.mean())/data.std()
    norm['Chance of Admit '] = data['Chance of Admit ']
    return norm


# In[3]:


data= pd.read_csv('AdmissionDataset/data.csv')
data=data.drop(columns='Serial No.')
list_attribute= data.columns[:-1]
# print list_attribute
data=normalize(data,list_attribute)
li=[1]*len(data.index)
data.insert(0, "Extra", li) 
train_data=data.sample(frac=0.8)
validation_data=data.drop(train_data.index)

validation_data=pd.read_csv(sys.argv[1])
validation_data.insert(0, "Extra", li) 
validation_data=validation_data.drop(columns='Serial No.')


# In[4]:


def calc_beta_value(train_data):
#     print train_data
    X=train_data.iloc[:,:-1].values
    X_transpose=X.T
#     print X.shape
    a=inv(num.dot(X_transpose,X))
    b=num.dot(a,X_transpose)
    beta=num.dot(b,train_data['Chance of Admit '].values)
#     print beta
    return beta

beta=calc_beta_value(train_data)
# print beta


# In[5]:


def calc_predication(beta,test_data):
#     print test_data.iloc[:,:-1].values.shape
#     print beta.shape
    predication=num.dot(test_data.iloc[:,:-1].values,beta)
    return predication


# In[6]:


predication_p1=calc_predication(beta,validation_data)
# print predication
print "R2 Score"
print r2_score(validation_data['Chance of Admit '].tolist(),predication_p1)


# # Part 2

# In[7]:


def gradiant_method_mean_sq_error(X,Y,beta,learning_rate,m):
#     print X.shape
    a=num.dot(X,beta)
#     print Y.shape
    z=num.subtract(a,Y)
#     print z.shape
    a=(learning_rate/m)*(num.dot(z.T,X))
#     print a.shape
    beta=num.subtract(beta,a.T)
#     print beta.shape
    return beta


# In[8]:


beta=num.array([0]*8)
X=train_data.iloc[:,:-1].values
Y=num.array(train_data['Chance of Admit '].tolist())
beta=beta.reshape(8,1)
Y=Y.reshape(360,1)
m=len(train_data.index)
for i in range(500):
    beta=gradiant_method_mean_sq_error(X,Y,beta,0.0125,m)
# print beta
predication=calc_predication(beta,validation_data)
# print predication,beta
print "Gradient MSE R2 Score"
print r2_score(validation_data['Chance of Admit '].tolist(),predication)


# In[9]:


def mean_squared_error(actual,predication):
    error=0
    for i in range(len(actual)):
        error = error +  ((actual[i]-predication[i])**2)
    return float(error/len(actual))
print "Error"
print mean_squared_error(validation_data['Chance of Admit '].tolist(),predication)


# In[10]:


def gradiant_method_mean_abs_error(X,Y,beta,learning_rate,m):
    a=num.dot(X,beta)
#     print Y.shape
    z=num.subtract(a,Y)
    z=num.divide(z,num.abs(z))
#     print z.shape
    a=(learning_rate/(2*m))*(num.dot(z.T,X))
#     print a.shape
    beta=num.subtract(beta,a.T)
#     print beta.shape
    return beta


# In[11]:


beta=num.array([0]*8)
X=train_data.iloc[:,:-1].values
Y=num.array(train_data['Chance of Admit '].tolist())
beta=beta.reshape(8,1)
Y=Y.reshape(360,1)
m=len(train_data.index)
for i in range(1000):
     beta=gradiant_method_mean_abs_error(X,Y,beta,0.0125,m)
# print beta
predication=calc_predication(beta,validation_data)
# print predication
print "Gradient MAE R2 Score"
print r2_score(validation_data['Chance of Admit '].tolist(),predication)


# In[12]:


def mean_absoulte_error(actual,predication):
    error=0
    for i in range(len(actual)):
        error= error + abs(actual[i]-predication[i])
    return float(error/len(actual))
print "Error"
print mean_absoulte_error(validation_data['Chance of Admit '].tolist(),predication)


# In[13]:


def gradiant_method_mean_precentage_error(X,Y,beta,learning_rate,m):
    a=num.dot(X,beta)
#     print Y.shape
    z=num.subtract(a,Y)
    z=num.divide(z,Y*num.abs(z))
#     print z.shape
    a=(learning_rate/(2*m))*(num.dot(z.T,X))
#     print a.shape
    beta=num.subtract(beta,a.T)
#     print beta.shape
    return beta


# In[14]:


beta=num.array([0]*8)
X=train_data.iloc[:,:-1].values
Y=num.array(train_data['Chance of Admit '].tolist())
beta=beta.reshape(8,1)
Y=Y.reshape(360,1)
m=len(train_data.index)
for i in range(1000):
    beta=gradiant_method_mean_precentage_error(X,Y,beta,0.0125,m)
# print beta
predication=calc_predication(beta,validation_data)
print "Gradient MAPE R2 Score"
print r2_score(validation_data['Chance of Admit '].tolist(),predication)


# In[15]:


def mean_absoulte_precentage_error(actual,predication):
    error=0
    for i in range(len(actual)):
        error=error +  abs((actual[i]-predication[i])/actual[i])
    return float(error*100/len(actual))
print "Error"
print mean_absoulte_precentage_error(validation_data['Chance of Admit '].tolist(),predication)


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
X=train_data.iloc[:,:-1].values
Y=train_data['Chance of Admit '].tolist()
regr.fit(X, Y)
pred = regr.predict(validation_data.iloc[:,:-1].values)
# print(regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(validation_data['Chance of Admit '].tolist(), pred))
print('Variance score: %f' % r2_score(validation_data['Chance of Admit '].tolist(), pred))


# In[17]:


print "MAE , MSE ,MAPE Error for Matrix Method Beta "
print mean_squared_error(validation_data['Chance of Admit '].tolist(),predication_p1)


# In[18]:


print mean_absoulte_error(validation_data['Chance of Admit '].tolist(),predication_p1)


# In[19]:


print mean_absoulte_precentage_error(validation_data['Chance of Admit '].tolist(),predication_p1)


# # Part 3

# In[20]:


print beta


# In[ ]:





# In[ ]:




