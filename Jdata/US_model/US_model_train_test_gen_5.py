# from basic_feat0518 import *
# from u_feat0518 import *
# from s_feat0518 import *
# from us_feat0518 import *

# 标签
def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s_cate==8.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[(actions['type'] == 4)&(actions['cate']==8)]
        actions =actions[['user_id','sku_id']].drop_duplicates(['user_id','sku_id']).reset_index()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
#         actions.to_csv(dump_path, index=False)
    return actions

# 训练集
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date):
    dump_path = './cache/train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        print ('================>>>train feature starting')
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=10)
        start_days = str(start_days.strftime('%Y-%m-%d'))
        actions = get_actions(start_days, train_end_date)
        actions = actions[actions['cate'] == 8][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])
        print (actions.shape)
        actions = pd.merge(actions, get_basic_user_feat(), on='user_id', how='left')
        print (actions.shape)
        start_days = "2016-02-01"
        print ('================>>>merge user feature')
#       for i in (1,2,3,5,7,10,15,21,30):
        for i in(1,2,3,7,14,28):
#             start_day1=datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
#             start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train user_feat')
            actions=pd.merge(actions,get_user_feat(train_start_date, train_end_date, i),on='user_id',how='left')
            print('=======>>>>train user_feat11')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat13')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_18')
            actions = pd.merge(actions, get_action_u0509_feat_18(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_21')
            actions = pd.merge(actions, get_action_u0509_feat_21(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_22')
            actions = pd.merge(actions, get_action_u0509_feat_22(train_start_date, train_end_date, i), on='user_id',how='left')
            print('=======>>>>train u0509_feat_23')
            actions = pd.merge(actions, get_action_u0509_feat_23(train_start_date, train_end_date, i), on='user_id',how='left')
            print('=======>>>>train u0509_feat_24')
            if(i<=10):
                actions = pd.merge(actions, get_action_u0509_feat_24(train_start_date, train_end_date, i), on='user_id',how='left')
                print('=======>>>>train u0509_feat_27')
            actions = pd.merge(actions, get_action_u0509_feat_27(train_start_date, train_end_date, i), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat1')
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat2')
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6')
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6_six')
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat7')
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8')
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8_2')
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat9')
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat10')
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat12')
#         actions = pd.merge(actions, get_action_user_feat12(start_days, train_end_date), on='user_id', how='left')
#         print (actions.shape)
        print('=======>>>>train user_feat14')
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat15')
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat16')
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_19')
        actions = pd.merge(actions, get_action_u0509_feat_19(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_20')
        actions = pd.merge(actions, get_action_u0509_feat_20(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_25')
        actions = pd.merge(actions, get_action_u0509_feat_25(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)


        print ('================>>>merge product feature')
        actions = pd.merge(actions, get_basic_product_feat(), on='sku_id', how='left')
        print (actions.shape)
        actions = pd.merge(actions, get_comments_product_feat(train_start_date, train_end_date), on='sku_id', how='left')
        print (actions.shape)
#        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):

        

        for i in(1,2,3,7,14,28):
            
            print('=======>>>>train p0509_feat')
            actions = pd.merge(actions, get_action_p0509_feat(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_11')
            actions = pd.merge(actions, get_action_product_feat_11(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_13')
            actions = pd.merge(actions, get_action_product_feat_13(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat_21')
            actions = pd.merge(actions, get_action_p0509_feat_21(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat_23')
            actions = pd.merge(actions, get_action_p0509_feat_23(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train p0509_feat24')
            if(i<=10):
                actions = pd.merge(actions, get_action_p0509_feat_24(train_start_date, train_end_date, i), on='sku_id', how='left')
                print('=======>>>>train p0509_feat27')
            actions = pd.merge(actions, get_action_p0509_feat_27(train_start_date, train_end_date, i), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_1')
        actions = pd.merge(actions, get_action_product_feat_1(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat2')
        actions = pd.merge(actions, get_action_p0509_feat_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat6')
        actions = pd.merge(actions, get_action_p0509_feat_6(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat_6_six')
        actions = pd.merge(actions, get_action_p0509_feat_6_six(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8')
        actions = pd.merge(actions, get_action_p0509_feat_8(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8_2')
        actions = pd.merge(actions, get_action_p0509_feat_8_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat9')
        actions = pd.merge(actions, get_action_p0509_feat_9(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat10')
        actions = pd.merge(actions, get_action_product_feat_10(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
#         print('=======>>>>train p0509_feat12')
#         actions = pd.merge(actions, get_action_product_feat_12(start_days, train_end_date), on='sku_id', how='left')
#         print (actions.shape)
        print('=======>>>>train product_feat_14')
        actions = pd.merge(actions, get_action_product_feat_14(train_start_date, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat15')
        actions = pd.merge(actions, get_action_p0509_feat_15(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_16')
        actions = pd.merge(actions, get_action_product_feat_16(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat17')
        actions = pd.merge(actions, get_action_p0509_feat_17(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat19')
        actions = pd.merge(actions, get_action_p0509_feat_19(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat23')

        print('=======>>>>train p0509_feat28')
        actions = pd.merge(actions, get_action_p0509_feat_28(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)

        print ('================>>>merge user_sku feature')
        print ('train get_accumulate_action_feat')
        actions = pd.merge(actions, get_accumulate_action_feat(train_start_date, train_end_date), how='left', on=['user_id', 'sku_id'])
        print ('train U_P_feat1')
        
        actions = pd.merge(actions, get_action_U_P_feat1(start_days, train_end_date), how="left",on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat3')
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days_2 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days_2 = start_days_2.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_action_feat(start_days_2, train_end_date), how='left',on=['user_id', 'sku_id'])
        actions.columns=['user_id','sku_id']+['action_feat'+str(i) for i in range(1,actions.shape[1]-1)]
    
        for i in(1,2,3,7,14,28):
            actions = pd.merge(actions, get_action_U_P_feat3(i, start_days, train_end_date), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat6( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat9( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            if(i<=10):
                actions=pd.merge(actions, get_action_U_P_feat_0509_feat_24( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
 
        print ('train U_P_feat4')
        actions = pd.merge(actions, get_action_U_P_feat4(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat5')
        actions = pd.merge(actions, get_action_U_P_feat5(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
#         print ('train U_P_feat7')
#         actions = pd.merge(actions, get_action_U_P_feat7(train_start_date, train_end_date), how="left",
#                                     on=['user_id', 'sku_id'])
#         print(actions.shape)
        print ('train U_P_feat8')
        actions = pd.merge(actions, get_action_U_P_feat8(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat14')
        actions = pd.merge(actions, get_action_U_P_feat14(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat16')
        actions = pd.merge(actions, get_action_U_P_feat16(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])

        print(actions.shape)
        print("train get_labels")
        actions=pd.merge(actions,get_labels(test_start_date, test_end_date),how='left',on=['user_id','sku_id'])
        print(actions.shape)
        actions = actions.fillna(0)
    return actions


# 测试集
def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/test_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        print ('================>>>train feature starting')
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=10)
        start_days = str(start_days.strftime('%Y-%m-%d'))
        actions = get_actions(start_days, train_end_date)
        actions = actions[actions['cate'] == 8][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])
        print (actions.shape)
        actions = pd.merge(actions, get_basic_user_feat(), on='user_id', how='left')
        print (actions.shape)
        start_days = "2016-02-01"
        print ('================>>>merge user feature')
    # for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
#        for i in (1, 2, 3, 7, 14, 28):
        for i in(1,2,3,7,14,28):
            start_day1 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train user_feat')
            actions = pd.merge(actions, get_user_feat(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat11')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train user_feat13')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), on='user_id', how='left')
            print('=======>>>>train u0509_feat_18')
            actions = pd.merge(actions, get_action_u0509_feat_18(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_21')
            actions = pd.merge(actions, get_action_u0509_feat_21(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_22')
            actions = pd.merge(actions, get_action_u0509_feat_22(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_23')
            actions = pd.merge(actions, get_action_u0509_feat_23(train_start_date, train_end_date, i), on='user_id',
                               how='left')
            print('=======>>>>train u0509_feat_24')
            if(i<=10):
                actions = pd.merge(actions, get_action_u0509_feat_24(train_start_date, train_end_date, i), on='user_id',
                                   how='left')
                print('=======>>>>train u0509_feat_27')
            actions = pd.merge(actions, get_action_u0509_feat_27(train_start_date, train_end_date, i), on='user_id',
                               how='left')
        print (actions.shape)
        print('=======>>>>train user_feat1')
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat2')
        actions = pd.merge(actions, get_action_user_feat2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6')
        actions = pd.merge(actions, get_action_user_feat6(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat6_six')
        actions = pd.merge(actions, get_action_user_feat6_six(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat7')
        actions = pd.merge(actions, get_action_user_feat7(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8')
        actions = pd.merge(actions, get_action_user_feat8(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat8_2')
        actions = pd.merge(actions, get_action_user_feat8_2(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat9')
        actions = pd.merge(actions, get_action_user_feat9(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat10')
        actions = pd.merge(actions, get_action_user_feat10(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat12')
#         actions = pd.merge(actions, get_action_user_feat12(start_days, train_end_date), on='user_id', how='left')
#         print (actions.shape)
        print('=======>>>>train user_feat14')
        actions = pd.merge(actions, get_action_user_feat14(train_start_date, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat15')
        actions = pd.merge(actions, get_action_user_feat15(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train user_feat16')
        actions = pd.merge(actions, get_action_user_feat16(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_19')
        actions = pd.merge(actions, get_action_u0509_feat_19(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_20')
        actions = pd.merge(actions, get_action_u0509_feat_20(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)
        print('=======>>>>train u0509_feat_25')
        actions = pd.merge(actions, get_action_u0509_feat_25(start_days, train_end_date), on='user_id', how='left')
        print (actions.shape)

        print ('================>>>merge product feature')
        actions = pd.merge(actions, get_basic_product_feat(), on='sku_id', how='left')
        print (actions.shape)
        actions = pd.merge(actions, get_comments_product_feat(train_start_date, train_end_date), on='sku_id',
                           how='left')
        print (actions.shape)
#         for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        for i in(1,2,3,7,14,28):
        
            start_day1 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_day1 = start_day1.strftime('%Y-%m-%d')
            print('=======>>>>train p0509_feat')
            actions = pd.merge(actions, get_action_p0509_feat(train_start_date, train_end_date, i), on='sku_id', how='left')
            print('=======>>>>train product_feat_11')
            actions = pd.merge(actions, get_action_product_feat_11(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train product_feat_13')
            actions = pd.merge(actions, get_action_product_feat_13(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat_21')
            actions = pd.merge(actions, get_action_p0509_feat_21(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat_23')
            actions = pd.merge(actions, get_action_p0509_feat_23(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat24')
            if(i<=10):
                actions = pd.merge(actions, get_action_p0509_feat_24(train_start_date, train_end_date, i), on='sku_id',
                               how='left')
            print('=======>>>>train p0509_feat27')
            actions = pd.merge(actions, get_action_p0509_feat_27(train_start_date, train_end_date,i), on='sku_id',
                               how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_1')
        actions = pd.merge(actions, get_action_product_feat_1(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat2')
        actions = pd.merge(actions, get_action_p0509_feat_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat6')
        actions = pd.merge(actions, get_action_p0509_feat_6(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat_6_six')
        actions = pd.merge(actions, get_action_p0509_feat_6_six(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8')
        actions = pd.merge(actions, get_action_p0509_feat_8(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat8_2')
        actions = pd.merge(actions, get_action_p0509_feat_8_2(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat9')
        actions = pd.merge(actions, get_action_p0509_feat_9(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat10')
        actions = pd.merge(actions, get_action_product_feat_10(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat12')
#         actions = pd.merge(actions, get_action_product_feat_12(start_days, train_end_date), on='sku_id', how='left')
#         print (actions.shape)
        print('=======>>>>train product_feat_14')
        actions = pd.merge(actions, get_action_product_feat_14(train_start_date, train_end_date), on='sku_id',
                           how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat15')
        actions = pd.merge(actions, get_action_p0509_feat_15(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train product_feat_16')
        actions = pd.merge(actions, get_action_product_feat_16(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat17')
        actions = pd.merge(actions, get_action_p0509_feat_17(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat19')
        actions = pd.merge(actions, get_action_p0509_feat_19(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)
        print('=======>>>>train p0509_feat23')

        print('=======>>>>train p0509_feat28')
        actions = pd.merge(actions, get_action_p0509_feat_28(start_days, train_end_date), on='sku_id', how='left')
        print (actions.shape)

        print ('================>>>merge user_sku feature')
        print ('train get_accumulate_action_feat')
        
        actions = pd.merge(actions, get_accumulate_action_feat(train_start_date, train_end_date), how='left',
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat1')
        actions = pd.merge(actions, get_action_U_P_feat1(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat3')
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days_2 = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days_2 = start_days_2.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_action_feat(start_days_2, train_end_date), how='left',on=['user_id', 'sku_id'])
        actions.columns=['user_id','sku_id']+['action_feat'+str(i) for i in range(1,actions.shape[1]-1)]
    
        for i in(1,2,3,7,14,28):
            actions = pd.merge(actions, get_action_U_P_feat3(i, start_days, train_end_date), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat6( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            actions=pd.merge(actions, get_action_U_P_feat9( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            if(i<=10):
                actions=pd.merge(actions, get_action_U_P_feat_0509_feat_24( train_start_date, train_end_date,i), on=['user_id', 'sku_id'],
                               how='left')
            
        print ('train U_P_feat4')
        actions = pd.merge(actions, get_action_U_P_feat4(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat5')
        actions = pd.merge(actions, get_action_U_P_feat5(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat7')
#         actions = pd.merge(actions, get_action_U_P_feat7(train_start_date, train_end_date), how="left",
#                            on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat8')
        actions = pd.merge(actions, get_action_U_P_feat8(start_days, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print(actions.shape)
        print ('train U_P_feat14')
        actions = pd.merge(actions, get_action_U_P_feat14(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])
        print ('train U_P_feat16')
        actions = pd.merge(actions, get_action_U_P_feat16(train_start_date, train_end_date), how="left",
                           on=['user_id', 'sku_id'])

        print(actions.shape)
        actions= actions.fillna(0)
    return actions


train_start_date = '2016-03-10'
train_end_date = '2016-04-11'
test_start_date = '2016-04-11'
test_end_date = '2016-04-16'

sub_start_date = '2016-03-15'
sub_end_date = '2016-04-16'

val_train_start_date = '2016-03-05'
val_train_end_date = '2016-04-06'
val_test_start_date = '2016-04-06'
val_test_end_date = '2016-04-11'
print ('=====================>>>>>>>>生成训练数据集')
actions = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
label_value = actions['label'].value_counts()
print ('训练集正负样本数分别为：'
       '正样本数为:'+str(label_value[1])+'负样本数为:'+str(label_value[0])+
       '正负样本比例为:'+str(1.0*label_value[1]/label_value[0]))

train,test=train_test_split(actions.values,test_size=0.2,random_state=0)
train=pd.DataFrame(train,columns=actions.columns)
test=pd.DataFrame(test,columns=actions.columns)

X_train=train.drop(['user_id','sku_id','label'],axis=1)
X_test=test.drop(['user_id','sku_id','label'],axis=1)
y_train=train[['label']]
y_test=test[['label']]
train_index=train[['user_id','sku_id']].copy()
test_index=test[['user_id','sku_id']].copy()

# print ('=====================>>>>>>>>生成验证数据集')
# actions_val = make_train_set(val_train_start_date, val_train_end_date, val_test_start_date, val_test_end_date)
# label_value = actions_val['label'].value_counts()
# print ('验证集正负样本数分别为:'
#        '正样本数为:'+str(label_value[1])+'负样本数为:'+str(label_value[0])+
#        '正负样本比例为:'+str(1.0*label_value[1]/label_value[0]))

# X_val=actions_val.drop(['user_id','sku_id','label'],axis=1)
# y_val=actions_val[['label']]
# val_index=actions_val[['user_id','sku_id']].copy()


print ('=====================>>>>>>>>生成测试数据集')
sub_test_data = make_test_set(sub_start_date, sub_end_date)
sub_trainning_data=sub_test_data.drop(['user_id','sku_id'],axis=1)
sub_user_index=sub_test_data[['user_id','sku_id']].copy()
print("finish")



print ('=====================>>>>>>>>生成验证数据集')
actions_val = make_train_set(val_train_start_date, val_train_end_date, val_test_start_date, val_test_end_date)
label_value = actions_val['label'].value_counts()
print ('验证集正负样本数分别为:'
       '正样本数为:'+str(label_value[1])+'负样本数为:'+str(label_value[0])+
       '正负样本比例为:'+str(1.0*label_value[1]/label_value[0]))

X_val=actions_val.drop(['user_id','sku_id','label'],axis=1)
y_val=actions_val[['label']]
val_index=actions_val[['user_id','sku_id']].copy()