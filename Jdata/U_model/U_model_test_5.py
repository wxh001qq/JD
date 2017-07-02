
label_val_value=val_actions['label'].value_counts()
print ('验证集正负样本数分别为：正样本数为'+str(label_val_value[1])+'负样本数为:'+str(label_val_value[0])+
       '正负样本比例为:'+str(1.0*label_val_value[1]/label_val_value[0]))

label_qeual_1 = val_actions[val_actions['label']==1][['user_id','label']]
print("验证集正样本数:"+str(label_qeual_1.shape))
print ('==========>>>train xgboost model ....')

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
# param = {'learning_rate' : 0.1,
#         'n_estimators': 1000,
#         'max_depth': 3,
#         'min_child_weight': 5,
#         'gamma': 0,
#         'subsample': 1.0,
#         'colsample_bytree': 0.8,
#         'eta': 0.05,
#         'silent': 1,
#         'objective':
#         'binary:logistic',
#         'scale_pos_weight':1}

param = {'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        'min_child_weight': 1.393849,
        'gamma': 0.001783,
         'gamma':2,
        'subsample': 0.537617,
        'colsample_bytree': 0.370318,
        'eta': 0.05,
        'silent': 1,
        'objective':
        'binary:logistic',
        'scale_pos_weight':1}


num_round =100
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=100)


print ('========================>>> val evaluation')

def evaluation(pred,label):
    print ('========>>>evaluation')
#     print("zlzl")
#     print(pred.head())
#     print(label.head())
#     print(pred.shape)
#     print(label.shape)
    actions = label    #['user_id',label]
    result = pred      #['user_id','prob']
    # 所有购买用户
    all_user_set = actions['user_id'].unique()
    # 所有品类中预测购买的用户
#     all_user_test_set = result['user_id'].unique()
    all_user_test_set = result['user_id']
    # 计算所有用户购买评价指标
    pos, neg = 0, 0
#     print("hn")
#     print(len(all_user_test_set))
#     print(len(all_user_set))

    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
            
#     print("zl")
    print("对的个数:"+str(pos))
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    print('F11=' + str(F11))


def val_evaluation(val_data,bst):
    index_val=val_data[['user_id']]
    X_val=val_data.drop(['user_id','label'],axis=1)
    y_val=val_data[['label']]
    X_val=xgb.DMatrix(X_val)
    y_val=pd.concat([index_val,y_val],axis=1)
    y_val=y_val[y_val['label']==1]
    y_predict=bst.predict(X_val)
    y_predict=pd.DataFrame(y_predict,columns=['prob'])
    y_predict=pd.concat([index_val,y_predict],axis=1)
    
    y_predict.sort_values(by=['user_id','prob'],ascending=[0,0],inplace=True)
    y_predict = y_predict.groupby('user_id').first().reset_index()
    y_predict.sort_values(by=['prob'],ascending=[0],inplace=True)

    
    print ('========>>>evalution result ')
    for i in (400,450,500,550,600,650,700,750,800,850,900,950,1000):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ('验证集前%s个结果的F11:'%i)
        y_predict_i=y_predict[:i]
        evaluation( y_predict_i,y_val)

print  ('========>>>验证集结果：')
val_evaluation(val_actions,bst)


print ('==========>>>print feature importance')
# score=bst.get_fscore()
# f_id = pd.DataFrame(list(score.keys()))
# f_score=pd.DataFrame(list(score.values()))
# fscore=pd.concat([f_id,f_score],axis=1)
# fscore.columns=['f_id','f_score']
# fscore.sort_values(by=['f_score'],ascending=[0],inplace=True)
# fscore.to_csv('./sub/u_feat_importace.csv',index=False)

# ============================================>>>>
print ('==========>>>predict test data label')


sub_trainning_data_1 = xgb.DMatrix(sub_trainning_data)
y = bst.predict(sub_trainning_data_1)
sub_user_index['label'] = y
pred=sub_user_index

pred.sort_values(by=['user_id','label'],ascending=[0,0],inplace=True)
pred = pred.groupby('user_id').first().reset_index()
result=pred.sort_values(by=['label'],ascending=[0])
result['user_id']=result['user_id'].astype('int')


name=str(datetime.now()).replace(':','-').split('.')[0]
result.to_csv('./sub/%s.csv'%name,index=False,index_label=False )