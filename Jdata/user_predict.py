
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import random
import xgboost as xgb


path = '../data'


feature_train = pd.read_csv(path + '/feature03/feature_train.csv')
feature_test = pd.read_csv(path + '/feature03/feature_test.csv')
feature_val = pd.read_csv(path + '/feature03/feature_val.csv')


def evaluation(answer,pre):
    a = (pd.merge(answer,pre,how='inner',on='user_id'))
    n11 = a.user_id.unique().shape[0]
    nda = answer.shape[0]
    npred = pre.shape[0]
    p11 = float(n11) / npred
    r11 = float(n11) / nda
    f11 = 6 * r11 * p11 / (5*r11 + p11)
    print("P11:",p11," R11:",r11," F11:",f11)
    print("score:", 0.4*f11)

def train(feature_train):
    pos = feature_train[feature_train['label']==1]
    l = len(pos)
    print('%d trainning samples with %d postive samples' %(len(feature_train), l))
    label = feature_train['label']
    x_train = feature_train.drop(['user_id', 'label'], axis=1)
    print('=====>Trainning...')
    clf = XGBClassifier(learning_rate=0.05,
                        n_estimators=450,
                        max_depth=3)
    clf.fit(x_train, label)
    print('Done!')
    return clf
    
def predict(feature_test, clf,threshold):
    x_test = feature_test.drop('user_id', axis=1)
    proba = pd.DataFrame(clf.predict_proba(x_test)[:, 1])
    result = pd.concat([feature_test, proba], axis=1)
    result.rename(columns={0: "proba"}, inplace=True)
    result = result[['user_id', 'proba']]
    result = result.sort_values(by='proba', ascending=False).drop_duplicates('user_id')
    result = result.iloc[0:threshold,:]
    result['sku_id'] = 1
    result = result[['user_id', 'sku_id']]
    return result

def validate(feature_val, real, clf):
	thresholds = np.arange(600,1200,50)
	for threshold in thresholds:		
	    preds = predict(feature_val, clf,threshold)
	    print(evaluation(real, preds))

#读取测试集label
real = pd.read_csv(path + '/feature01/label_val.csv')
real.shape

#训练模型
clf = train(feature_train)

clf.feature_importances_

#线下验证
validate(feature_val, real, clf)

#线上
result = predict(feature_test, clf)
result.to_csv(path + '/user.csv', index=False)



