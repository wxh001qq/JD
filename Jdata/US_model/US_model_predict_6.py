#from gen_data0518 import *
import xgboost as xgb

print ('start running ....')

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
param = {'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'silent': 1,
        'objective':
        'binary:logistic',
        'scale_pos_weight':1}

num_round =150
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=100)

print ('========================>>> val evaluation')
def evaluation(pred,label):
    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    # 所有品类中预测购买的用户商品对
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0, 0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print("U模型对的个数"+str(pos))
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    
    print("US模型对的个数"+str(pos))
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print ('F12=' + str(F12))
    print ('score=' + str(score))


def val_evaluation(test_index,X_test,y_test,bst):
    X_test=xgb.DMatrix(X_test)
#     y_test=pd.concat([test_index,y_test],axis=1)
    print("11111")
    print(y_test.head())
    y_test=y_test[y_test['label']==1]
    print(y_test.head())
    val=bst.predict(X_test)
#    print(len(val))
#    print(test_index.shape)
    val=pd.DataFrame(val,columns=['prob'])
    val=pd.concat([test_index,val],axis=1)
#     P = list(get_basic_product_feat()['sku_id'])
#     for term in P:
#         map_p[term]=1
#     val = val[val['sku_id'].map(lambda x: True if x in map_p else False)]
    
    P=get_basic_product_feat()[['sku_id']]
    P['sku_label']=1
    val=pd.merge(val,P,on='sku_id',how='left')
    print(val.head())
    val=val[val['sku_label']==1][['user_id','sku_id','prob']]

    
    val.sort_values(by=['user_id', 'prob'], ascending=[0, 0], inplace=True)
    val = val.groupby('user_id').first().reset_index()
    val.sort_values(by=['prob'], ascending=[0], inplace=True)
#     val=val[val['prob']>0.05]
#     print("11111")
#     print(val.head())
#     print(y_test.head())
    for i in range(500,1000,50):
        print("+++++++++++++++++++++++++++++++++++++++++++")
        print("数据取前top:"+str(i))
        val_i = val[:i]
        evaluation(val_i, y_test)

#  get_labels()
val_test_start_date = '2016-04-06'
val_test_end_date = '2016-04-11'   
# print(y_val)
y_val_real=get_labels(val_test_start_date,val_test_end_date)
# print(y_val_real)
val_evaluation(val_index,X_val,y_val_real,bst)



#=========================================>>>>
# print ('==========>>>print feature importance')
# score=bst.get_fscore()
# print(score)
# f_id = pd.DataFrame(list(score.keys()))
# f_score=pd.DataFrame(list(score.values()))
# fscore=pd.concat([f_id,f_score],axis=1)
# fscore.columns=['f_id','f_score']
# fscore.sort_values(by=['f_score'],ascending=[0],inplace=True)
# fscore.to_csv('./sub/feat_importace.csv',index=False)

#============================================>>>>
print ('==========>>>predict test data label')
sub_trainning_data_1 = xgb.DMatrix(sub_trainning_data)
y = bst.predict(sub_trainning_data_1)
sub_user_index['label'] = y
print ('==========>>>finish test data label')
P=get_basic_product_feat()[['sku_id']]
P['sku_label']=1
pred=pd.merge(sub_user_index,P,on='sku_id',how='left')
pred=pred[pred['sku_label']==1][['user_id','sku_id','label']]

pred.sort_values(by=['user_id','label'],ascending=[0,0],inplace=True)
pred = pred.groupby('user_id').first().reset_index()
result=pred.sort_values(by=['label'],ascending=[0])
# pred = pred[pred['label'] >= 0.03]
# pred = pred[['user_id', 'sku_id']]
result['user_id']=result['user_id'].astype('int')

result=result[['user_id','sku_id']]

name=str(datetime.now()).replace(':','-').split('.')[0]
result.to_csv('./sub/%s.csv'%name,index=False,index_label=False )
print("finish")

