from sklearn.model_selection import train_test_split
import xgboost as xgb
train_start_date = '2016-03-10'
train_end_date = '2016-04-11'
test_start_date = '2016-04-11'
test_end_date = '2016-04-16'

sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'

user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)

X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
# #     dtrain=xgb.DMatrix(X_train, label=y_train)
# #     dtest=xgb.DMatrix(X_test, label=y_test)

# #     param = {'learning_rate' : 0.05, 'n_estimators': 1000, 'max_depth': 3,'min_child_weight': 5, 'gamma': 0,
# #               'subsample': 1.0, 'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
# #     num_round = 300
# #     param['nthread'] = 4
#     #param['eval_metric'] = "auc"
#      list(X2_X5_MAPPINGS.keys())
#     plst = param.items()
#     plst =list(param.keys())
# #     d = dict1.copy()
# # d.update(dict2)
#     plst += [('eval_metric', 'logloss')]

#     evallist = [(dtest, 'eval'), (dtrain, 'train')]

#     bst=xgb.train(plst, dtrain, num_round, evallist)
#     sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date,)
#     sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
#     y = bst.predict(sub_trainning_data)

xgb_1=xgb.XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3,min_child_weight= 5,
                        gamma=0,subsample=1.0, colsample_bytree= 0.8,
                        scale_pos_weight= 1, silent=True,objective= 'binary:logistic',nthread=8)
# nthread=10
# reg:logistic
xgb_1.fit(X_train, y_train,eval_set=[(X_train, y_train),(X_test,y_test)],early_stopping_rounds =300,verbose=True)

sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date,)


y= xgb_1.predict(sub_trainning_data.values)

#     sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date,)

sub_user_index['label'] = y
pred = sub_user_index[sub_user_index['label'] >= 0.03]
pred = pred[['user_id', 'sku_id']]
pred = pred.groupby('user_id').first().reset_index()
pred['user_id'] = pred['user_id'].astype(int)
pred.to_csv('./data/sub/submission4.csv', index=False, index_label=False)