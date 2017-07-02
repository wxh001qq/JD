#!/usr/bin/python
#-*-coding:utf-8-*-
import numpy as np
import xgboost as xgb
# from user_feat import *
from sklearn.model_selection import train_test_split


train_start_date = '2016-03-10'
train_end_date = '2016-04-11'
test_start_date = '2016-04-11'
test_end_date = '2016-04-16'

sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'

#训练数据集
actions = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date,user,ful_action,sel_action)
# print(np.isinf(actions))
# print(np.isnan(actions))


feature_name = actions.columns.values

# for index in feature_name[1:-1]:
#     actions["r"+index]=actions[index].rank(method='max')/actions.shape[0]

print(actions.shape)
actions_pos = actions[actions['label']==1]
actions_neg =  actions[actions['label']==0]

## train_neg,test_neg=train_test_split(actions_neg.values,test_size=0.15,random_state=0)

# #test_neg = pd.DataFrame(test_neg,columns=actions_neg.columns)

# #actions=pd.concat([actions_pos,test_neg])

actions_pos= pd.concat([actions_pos,actions_pos])
actions_pos= pd.concat([actions_pos,actions_pos])
actions_pos= pd.concat([actions_pos,actions_pos])
actions_pos= pd.concat([actions_pos,actions_pos])
actions=pd.concat([actions_pos,actions_neg])
print("+++++++++++++++++++++++")

label_value = actions['label'].value_counts()
print(label_value)
print ('训练集正负样本数分别为：正样本数为'+str(label_value[1])+'负样本数为:'+str(label_value[0])+
       '正负样本比例为:'+str(1.0*label_value[1]/label_value[0]))

train,test=train_test_split(actions.values,test_size=0.2,random_state=0)
train=pd.DataFrame(train,columns=actions.columns)
test=pd.DataFrame(test,columns=actions.columns)

X_train=train.drop(['user_id','label'],axis=1)
X_test=test.drop(['user_id','label'],axis=1)
y_train=train[['label']]
y_test=test[['label']]
train_index=train[['user_id']].copy()
test_index=test[['user_id']].copy()



# train_start_date = '2016-03-05'
# train_end_date = '2016-04-06'
# test_start_date = '2016-04-06'
# test_end_date = '2016-04-11'
# 验证集
val_train_start_date='2016-03-05'
val_train_end_date='2016-04-06'
val_test_start_date='2016-04-06'
val_test_end_date='2016-04-11'
val_actions = make_train_set(val_train_start_date, val_train_end_date, val_test_start_date, val_test_end_date,user,ful_action,sel_action)

feature_name = val_actions.columns.values

# for index in feature_name[1:-1]:
#     val_actions["r"+index]=val_actions[index].rank(method='max')/val_actions.shape[0]

print(val_actions.shape)

#测试数据集
sub_test_data = make_test_set(sub_start_date, sub_end_date,user,ful_action,sel_action)

feature_name = sub_test_data.columns.values
# for index in feature_name[1:]:
#     sub_test_data["r"+index]=sub_test_data[index].rank(method='max')/sub_test_data.shape[0]


sub_trainning_data=sub_test_data.drop(['user_id'],axis=1)
sub_user_index=sub_test_data[['user_id']].copy()    
    
print(sub_test_data.shape)
print("finish")

########################################################################
