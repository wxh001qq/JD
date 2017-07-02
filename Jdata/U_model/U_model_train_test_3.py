# 测试集
# ful_action = pd.read_csv('./data/JData_Action.csv', parse_dates=[2], infer_datetime_format=True)
# sel_action = pd.read_csv('./data/JData_subset_action.csv', parse_dates=[2, 7], infer_datetime_format=True)
def make_test_set(train_start_date, train_end_date,user,ful_action,sel_action):
    dump_path = './cache/u_test_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=10)).split(' ')[0]
        actions = get_actions(start_days, train_end_date)
        actions=actions[actions['cate']==8][['user_id']].drop_duplicates(['user_id'])
        print (actions.shape)
#         start_days = train_start_date
        start_days = "2016-02-01"
#         actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
#         print(actions.shape)
#         
     
#         actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
#         print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat5(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat12(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_u0513_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, user_features(user,ful_action,sel_action,train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_1(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_2(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        
        
        for i in (1, 2, 3, 7, 14, 28):
            actions = pd.merge(actions, get_action_user_feat_six_xingwei(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0509_1_30(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_3(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_feat(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_4(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_u0515_feat5(train_start_date, train_end_date,i), how='left', on='user_id')
            if(i<=10):
                actions = pd.merge(actions,get_action_user_feat0509_1_31(train_start_date, train_end_date,i), how='left', on='user_id')
           
        print(actions.shape)
        print(actions.shape)

        actions = actions.fillna(0)
#         user_id = actions[['user_id']]
#         del actions['user_id']
#         actions = actions.fillna(0)
#         actions=actions.replace(np.inf,0)
# #         print(actions.head())
#         columns = actions.columns

#         min_max_scale = preprocessing.MinMaxScaler()
#         actions=actions.replace(np.inf,0)
#         actions = min_max_scale.fit_transform(actions.values)
#         actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
        actions.to_csv(dump_path,index=False)
    return actions


# 训练集
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date,user,ful_action,sel_action):
    dump_path = './cache/zlzlzl_u_train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=10)).split(' ')[0]
        actions = get_actions(start_days, train_end_date)
        actions=actions[actions['cate']==8][['user_id']].drop_duplicates(['user_id'])
        print (actions.shape)
#         start_days = train_start_date
        start_days = "2016-02-01"
#         actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
        print(actions.shape)
        
#         actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
#         print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat5(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat12(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_u0513_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, user_features(user,ful_action,sel_action,train_end_date), how='left', on='user_id')
        print (actions.shape)
        
        actions = pd.merge(actions, get_action_user_feat0515_2_1(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        actions = pd.merge(actions, get_action_user_feat0515_2_2(train_start_date, train_end_date), how='left', on='user_id')
        print (actions.shape)
        
        for i in (1, 2, 3,7, 14, 28):
            actions = pd.merge(actions, get_action_user_feat_six_xingwei(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0509_1_30(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_3(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_feat(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_user_feat0515_2_4(train_start_date, train_end_date,i), how='left', on='user_id')
            actions = pd.merge(actions, get_action_u0515_feat5(train_start_date, train_end_date,i), how='left', on='user_id')
            if(i<=10):
                actions = pd.merge(actions,get_action_user_feat0509_1_31(train_start_date, train_end_date,i), how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_user_labels(test_start_date, test_end_date), how='left', on='user_id')
        
        actions = actions.fillna(0)
        print(actions.shape)
#         user_id = actions[['user_id']]
#         del actions['user_id']
#         actions = actions.fillna(0)
#         actions=actions.replace(np.inf,0)
# #         print(actions.head())
#         columns = actions.columns

#         min_max_scale = preprocessing.MinMaxScaler()
#         actions=actions.replace(np.inf,0)
#         actions = min_max_scale.fit_transform(actions.values)
#         actions = pd.concat([user_id, pd.DataFrame(actions,columns = columns)], axis=1)
        actions.to_csv(dump_path,index=False)
    return  actions

print("finish")






###########################################################################################
