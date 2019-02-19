#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as num
import pandas as pd
import math
from pprint import pprint
import sys


# In[11]:


data =pd.read_csv('LoanDataset/data.csv',header=None)
list_columns=["Age","Experience","Annual Income" ,"ZIPCode","Family size","Avgerage spending per month" 
,"Education Level","Mortgage Value of house", "Output","securities account","certificate of deposit (CD)",
"internet banking","credit card issued"]
data=data[1:]
data.columns=["ID"]+list_columns
data=data.drop(columns="ID")
train_data=data.sample(frac=0.8,random_state=200)
validation_data=data.drop(train_data.index)
# list_columns.remove('ZIPCode')
validation_data =pd.read_csv(sys.argv[1],header=None)
# validation_data=validation_data[1:]
validation_data.columns=["ID"]+list_columns
validation_data=validation_data.drop(columns="ID")

categorial_data=list_columns[6:7]+list_columns[9:]
# print categorial_data
list_columns.remove('ZIPCode')
numerical_data=list(filter(lambda i:i not in categorial_data and i!= 'Output',list_columns))
# print numerical_data
# print list_columns


# In[12]:


numerical_data_std={}
numerical_data_mean={}
temp_0=train_data[train_data['Output']==0]
temp_1=train_data[train_data['Output']==1]
for i in numerical_data:
#     temp_0=train_data[output==1]a
    numerical_data_mean[i]={}
    numerical_data_std[i]={}
    numerical_data_mean[i][0]=temp_0[i].mean()
    numerical_data_std[i][0]=temp_0[i].std()
    numerical_data_mean[i][1]=temp_1[i].mean()
    numerical_data_std[i][1]=temp_1[i].std()
    
# pprint (numerical_data_mean)
# pprint (numerical_data_std)


# In[13]:


categorial_data_probability={}
temp_0=train_data[train_data['Output']==0]
temp_1=train_data[train_data['Output']==1]
for i in categorial_data:
#     temp_0=train_data[output==1]a
    categorial_data_probability[i]={}
    len_data_0=float(len(temp_0))
    len_data_1=float(len(temp_1))
#     print len_data_0,len_data_1
    for j in train_data[i].unique():
        categorial_data_probability[i][j]={}
        categorial_data_probability[i][j][0]=len(temp_0[temp_0[i]==j])/len_data_0
        categorial_data_probability[i][j][1]=len(temp_1[temp_1[i]==j])/len_data_1
# pprint (categorial_data_probability)
        
probability_zero=len(temp_0)/float(len(train_data))
probability_one=len(temp_1)/float(len(train_data))
# print probability_one
# print probability_zero


# In[14]:


# probability_one,probability_zero=1.0,1.0
# def calculation_output_data():
#     temp_0=train_data[train_data['Output']==0]
#     temp_1=train_data[train_data['Output']==1]
#     probability_zero=len(temp_0)/float(len(train_data))
#     probability_one=len(temp_1)/float(len(train_data))
# calculation_output_data()


# In[15]:


import math
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


# In[16]:


def predication(testing_data):
    predication_value_list=[]
    for testing_index, testing_row in testing_data.iterrows():
        pre_1=1
        pre_0=1
        for attr in list_columns:
            attr_value=testing_row[attr]
            if attr in categorial_data:
                pre_1=pre_1*categorial_data_probability[attr][attr_value][1]
                pre_0=pre_0*categorial_data_probability[attr][attr_value][0]
            elif attr in numerical_data:
                pre_1=pre_1*normpdf(attr_value,numerical_data_mean[attr][1],numerical_data_std[attr][1])
                pre_0=pre_0*normpdf(attr_value,numerical_data_mean[attr][0],numerical_data_std[attr][0])
            elif attr=='Output':
                pre_1=probability_one*pre_1
                pre_0=probability_zero*pre_0
            else :
                print "Something went Worng ",attr
        if pre_1>pre_0:
            predication_value_list.append(1)
        else:
            predication_value_list.append(0)
    return predication_value_list


# In[17]:


def calc_preformance(target_value,pridected_value):
    t_p=0
    f_p=0
    t_n=0
    f_n=0
    for i in range(len(target_value)):
        if target_value[i]==0 and target_value[i]==pridected_value[i]:
            t_n=t_n+1
        elif target_value[i]==1 and target_value[i]==pridected_value[i]:
            t_p=t_p+1
        elif pridected_value[i]==1 and target_value[i]==0:
            f_p=f_p+1
        elif pridected_value[i]==0 and target_value[i]==1:
            f_n=f_n+1
    if t_p!=0:
        accuracy=(t_n+t_p)/float(t_n+t_p+f_p+f_n)

        precision=(t_p)/float(t_p+f_p)
        recall=(t_p)/float(t_p+f_n)
        a=1/precision
        b=1/recall
        f1_score=2/(a+b)
    else :
        accuracy=0
        precision=0
        recall=0
        f1_score=0
#     print "ture positive",t_p
#     print "false positive",f_p
#     print "false negative",f_n
#     print "ture negative",t_n
    
    print "Accuracy ",accuracy
    print "Precision ",precision
    print "Recall ",recall
    print "F1 Score",f1_score
    return accuracy


# In[18]:


actual_value=validation_data['Output'].tolist()
predication_value=predication(validation_data)
ac=calc_preformance(actual_value,predication_value)


# In[ ]:





# In[ ]:




