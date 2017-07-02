import pandas as pd
import numpy as np
import datetime
import math

#%%
def get_features(ful_action, sel_action, user):
    sep_day = datetime.datetime(2016, 4, 6)
    predict_period = datetime.timedelta(5, 0)
    day = datetime.timedelta(1, 0)

    print('==========> 提取用户特征...')  
    for k in range(3):
        print('%d / 3' %(k+1))
        user = user.drop_duplicates('user_id')
        feature_i = None 
        for i in (1,3,5,15,30,'history'):
            if i== 'history':
                action = sel_action[sel_action['time']<sep_day]
                action_ful = ful_action[ful_action['time']<sep_day]
            else:
                action = sel_action[(sel_action['time']>=sep_day-i*day) & (sel_action['time']<sep_day)]
                action_ful = ful_action[(ful_action['time']>=sep_day-i*day) & (ful_action['time']<sep_day)]
            if feature_i is None:
                feature_i = user_features(user,action_ful, action,sep_day,str(i))
            else:
                feature_i = pd.merge(feature_i, user_features(user,action_ful ,action ,sep_day,str(i)), how='left', on=['user_id', 'user_lv_cd', 'reg_duration', 'reg_duration_cate'])
        feature_i['user_action_num_5/history'] = feature_i['u_action_num_5']/(feature_i['u_action_num_history']+0.001)
        feature_i['u_action_num_difference_1'] = feature_i['u_action_num_1']-feature_i['u_action_num_per_day_history']
        feature_i['u_browse_num_difference_1'] = feature_i['u_browse_num_1'] - feature_i['u_browse_num_per_day_history']
        feature_i['u_add_num_difference_1'] = feature_i['u_add_num_1'] - feature_i['u_add_num_per_day_history']
        feature_i['u_del_num_difference_1'] = feature_i['u_del_num_1'] - feature_i['u_del_num_per_day_history']
        feature_i['u_buy_num_difference_1'] = feature_i['u_buy_num_1'] - feature_i['u_buy_num_per_day_history']
        feature_i['u_follow_num_difference_1'] = feature_i['u_follow_num_1'] - feature_i['u_follow_num_per_day_history']
        feature_i['u_click_num_difference_1'] = feature_i['u_click_num_1'] - feature_i['u_click_num_per_day_history']

        if k<2:
            label_wnd = sel_action[(sel_action['time']>=sep_day) & (sel_action['time']<(sep_day+predict_period))]
            label = label_wnd[label_wnd['type']==4]
            label['buy'] = 1
            label = label[['user_id', 'buy']]
            label = label.groupby('user_id').sum().reset_index()
            label['label'] = np.where(label['buy']>0, 1, 0)
            label = label[['user_id', 'label']]
            feature_i = pd.merge(feature_i, label, how='left', on='user_id').fillna(0)
            if k==0:
                feature_i.to_csv('../../data/feature03/feature_train1.csv', index=False)
            else:
                feature_i.drop('label', axis=1).to_csv('../../data/feature03/feature_val.csv', index=False)
        else:
            feature_i.to_csv('../../data/feature03/feature_test1.csv', index=False)        
        sep_day += predict_period
    print('==========> Done')
#%%
def user_features(user, action_ful, action, sep_day,gap):
    print('=====> 提取特征...')    

    print('-- '+str(gap))
    #--------------- 子集特征 -------------------
    action_dummy = pd.get_dummies(action['type'], prefix='type')

    a=action_dummy.columns.values
    key = []
    true = [1,2,3,4,5,6]
    for i in range(len(a)):
        key.append(int(a[i][5]))
    for i in true:
        if i not in key:
            action_dummy['type_'+str(i)] = 0
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature.rename(columns={'type_0': 'u_action_num_'+str(gap),
                                      'type_1': 'u_browse_num_'+str(gap),
                                      'type_2': 'u_add_num_'+str(gap),
                                      'type_3': 'u_del_num_'+str(gap),
                                      'type_4': 'u_buy_num_'+str(gap),
                                      'type_5': 'u_follow_num_'+str(gap),
                                      'type_6': 'u_click_num_'+str(gap)}, inplace=True)
    
    # 转换率
    u_feature['u_buy/browse_num_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_browse_num_'+str(gap)]+0.001)*100
    u_feature['u_buy/add_num_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_add_num_'+str(gap)]+0.001)*100
    u_feature['u_buy/click_num_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    u_feature['u_buy/follow_num_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_follow_num_'+str(gap)]+0.001)*100
    u_feature['u_del/add_num_'+str(gap)] = u_feature['u_del_num_'+str(gap)]/(u_feature['u_add_num_'+str(gap)]+0.001)*100

    # 用户对商品行为
    action_sku = action_dummy.groupby(['user_id', 'sku_id']).sum().reset_index()
    action_sku = action_sku.drop('sku_id', axis=1)
    action_avg = action_sku.groupby('user_id').mean().reset_index()
    action_avg.rename(columns={'type_0': 'u_action_num_avg_'+str(gap),
                               'type_1': 'u_browse_num_avg_'+str(gap),
                               'type_2': 'u_add_num_avg_'+str(gap),
                               'type_3': 'u_del_num_avg_'+str(gap),
                               'type_4': 'u_buy_num_avg_'+str(gap),
                               'type_5': 'u_follow_num_avg_'+str(gap),
                               'type_6': 'u_click_num_avg_'+str(gap)}, inplace=True)
    action_median = action_sku.groupby('user_id').median().reset_index()
    action_median.rename(columns={'type_0': 'u_action_num_median_'+str(gap),
                               'type_1': 'u_browse_num_median_'+str(gap),
                               'type_2': 'u_add_num_median_'+str(gap),
                               'type_3': 'u_del_num_median_'+str(gap),
                               'type_4': 'u_buy_num_median_'+str(gap),
                               'type_5': 'u_follow_num_median_'+str(gap),
                               'type_6': 'u_click_num_median_'+str(gap)}, inplace=True)                           
    action_max = action_sku.groupby('user_id').max().reset_index()
    action_max.rename(columns={'type_0': 'u_action_num_max_'+str(gap),
                               'type_1': 'u_browse_num_max_'+str(gap),
                               'type_2': 'u_add_num_max_'+str(gap),
                               'type_3': 'u_del_num_max_'+str(gap),
                               'type_4': 'u_buy_num_max_'+str(gap),
                               'type_5': 'u_follow_num_max_'+str(gap),
                               'type_6': 'u_click_num_max_'+str(gap)}, inplace=True)
    u_feature = pd.merge(u_feature, action_avg, on='user_id', how='left')
    u_feature = pd.merge(u_feature, action_median, on='user_id', how='left')
    u_feature = pd.merge(u_feature, action_max, on='user_id', how='left').fillna(0)

    # 行为天数
    if gap != str(1):
        action_days = action[['user_id', 'date']]
        action_days = action_days.drop_duplicates()
        action_days = action_days.groupby('user_id').count().reset_index()
        action_days.rename(columns={'date': 'u_action_days_'+str(gap)}, inplace=True)
        u_feature = pd.merge(u_feature, action_days, on='user_id', how='left').fillna(0)
    
    # 时间特征
    action_days = action[['user_id', 'time']]
    action_start = action_days.groupby('user_id').min().reset_index()
    action_start.rename(columns={'time': 'start'}, inplace=True)
    action_end = action_days.groupby('user_id').max().reset_index()
    action_end.rename(columns={'time': 'end'}, inplace=True)
    action_duration = pd.merge(action_start, action_end, on='user_id')
    action_duration['u_action_attentionday_'+str(gap)] = sep_day - action_duration['start']
    action_duration['u_action_attentionday_'+str(gap)] = action_duration['u_action_attentionday_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration['u_action_duration_'+str(gap)] = action_duration['end'] - action_duration['start']
    action_duration['u_action_duration_'+str(gap)] = action_duration['u_action_duration_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration['u_silence_duration_'+str(gap)] = sep_day - action_duration['end']
    action_duration['u_silence_duration_'+str(gap)]= action_duration['u_silence_duration_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration = action_duration[['user_id','u_action_attentionday_'+str(gap) ,'u_action_duration_'+str(gap),'u_silence_duration_'+str(gap)]]
    u_feature = pd.merge(u_feature, action_duration, on='user_id', how='left').fillna(0)
    
    # 平均行为
    u_feature['u_action_num_per_day_'+str(gap)] = u_feature['u_action_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_browse_num_per_day_'+str(gap)] = u_feature['u_browse_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_add_num_per_day_'+str(gap)] = u_feature['u_add_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_del_num_per_day_'+str(gap)] = u_feature['u_del_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_buy_num_per_day_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_follow_num_per_day_'+str(gap)] = u_feature['u_follow_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    u_feature['u_click_num_per_day_'+str(gap)] = u_feature['u_click_num_'+str(gap)]/(u_feature['u_action_duration_'+str(gap)]+0.001)
    
    # 点击模块 (14, 21, 28, 110, 210, 216, 217)
    click_history = action[action['type']==6].fillna(-1)
    click_history = click_history[['user_id', 'model_id']]
    click_history['u_click14_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==14))
    click_history['u_click21_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==21))
    click_history['u_click28_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==28))
    click_history['u_click110_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==110))
    click_history['u_click210_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==210))
    click_history = click_history.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
    u_feature = pd.merge(u_feature, click_history, on='user_id', how='left')
    
    # 点击模块、点击总量
    u_feature['u_click14/click_sum_'+str(gap)] = u_feature['u_click14_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    u_feature['u_click21/click_sum_'+str(gap)] = u_feature['u_click21_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    u_feature['u_click28/click_sum_'+str(gap)] = u_feature['u_click28_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    u_feature['u_click110/click_sum_'+str(gap)] = u_feature['u_click110_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    u_feature['u_click210/click_sum_'+str(gap)] = u_feature['u_click210_'+str(gap)]/(u_feature['u_click_num_'+str(gap)]+0.001)*100
    
    # 商品特征
    user_sku = action_dummy.drop('user_id', axis=1).groupby('sku_id').sum().reset_index()
    user_sku = pd.merge(action_dummy[['user_id', 'sku_id']].drop_duplicates(), user_sku, on='sku_id', how='left')
    user_sku = user_sku.drop('sku_id', axis=1)
    user_sku_avg = user_sku.groupby('user_id').mean().reset_index()
    user_sku_avg.rename(columns={'type_0': 'u_sku_action_num_avg_'+str(gap),
                                 'type_1': 'u_sku_browse_num_avg_'+str(gap),
                                 'type_2': 'u_sku_add_num_avg_'+str(gap),
                                 'type_3': 'u_sku_del_num_avg_'+str(gap),
                                 'type_4': 'u_sku_buy_num_avg_'+str(gap),
                                 'type_5': 'u_sku_follow_num_avg_'+str(gap),
                                 'type_6': 'u_sku_click_num_avg_'+str(gap)}, inplace=True)
    user_sku_min = user_sku.groupby('user_id').min().reset_index()
    user_sku_min.rename(columns={'type_0': 'u_sku_action_num_min_'+str(gap),
                                 'type_1': 'u_sku_browse_num_min_'+str(gap),
                                 'type_2': 'u_sku_add_num_min_'+str(gap),
                                 'type_3': 'u_sku_del_num_min_'+str(gap),
                                 'type_4': 'u_sku_buy_num_min_'+str(gap),
                                 'type_5': 'u_sku_follow_num_min_'+str(gap),
                                 'type_6': 'u_sku_click_num_min_'+str(gap)}, inplace=True)
    u_feature = pd.merge(u_feature, user_sku_avg, on='user_id', how='left')
    u_feature = pd.merge(u_feature, user_sku_min, on='user_id', how='left').fillna(0)

    #--------------- 全集特征 -------------------
    action_dummy = pd.get_dummies(action_ful['type'], prefix='type')
    a=action_dummy.columns.values
    key = []
    true = [1,2,3,4,5,6]
    for i in range(len(a)):
        key.append(int(a[i][5]))
    for i in true:
        if i not in key:
            action_dummy['type_'+str(i)] = 0
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_ful[['user_id', 'sku_id']], action_dummy], axis=1)    
    u_feature_ful_history = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_ful_history.rename(columns={'type_0': 'u_action_num_ful_'+str(gap),
                                          'type_1': 'u_browse_num_ful_'+str(gap),
                                          'type_2': 'u_add_num_ful_'+str(gap),
                                          'type_3': 'u_del_num_ful_'+str(gap),
                                          'type_4': 'u_buy_num_ful_'+str(gap),
                                          'type_5': 'u_follow_num_ful_'+str(gap),
                                          'type_6': 'u_click_num_ful_'+str(gap)}, inplace=True)
    u_feature = pd.merge(u_feature, u_feature_ful_history, on='user_id', how='left')
    
    # 转换率
    u_feature['u_buy_ful/browse_num_ful'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_browse_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_buy_ful/add_num_ful'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_add_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_buy_ful/click_num_'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_buy_ful/follow_num_'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_follow_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_del_ful/add_num_'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_add_num_ful_'+str(gap)]+0.001)*100

    # 用户对商品行为
    action_sku = action_dummy.groupby(['user_id', 'sku_id']).sum().reset_index()
    action_sku = action_sku.drop('sku_id', axis=1)
    action_avg = action_sku.groupby('user_id').mean().reset_index()
    action_avg.rename(columns={'type_0': 'u_action_num_ful_avg_'+str(gap),
                               'type_1': 'u_browse_num_ful_avg_'+str(gap),
                               'type_2': 'u_add_num_ful_avg_'+str(gap),
                               'type_3': 'u_del_num_ful_avg_'+str(gap),
                               'type_4': 'u_buy_num_ful_avg_'+str(gap),
                               'type_5': 'u_follow_num_ful_avg_'+str(gap),
                               'type_6': 'u_click_num_ful_avg_'+str(gap)}, inplace=True)
    action_median = action_sku.groupby('user_id').median().reset_index()
    action_median.rename(columns={'type_0': 'u_action_num_ful_median_'+str(gap),
                               'type_1': 'u_browse_num_ful_median_'+str(gap),
                               'type_2': 'u_add_num_ful_median_'+str(gap),
                               'type_3': 'u_del_num_ful_median_'+str(gap),
                               'type_4': 'u_buy_num_ful_median_'+str(gap),
                               'type_5': 'u_follow_num_ful_median_'+str(gap),
                               'type_6': 'u_click_num_ful_median_'+str(gap)}, inplace=True)                           
    action_max = action_sku.groupby('user_id').max().reset_index()
    action_max.rename(columns={'type_0': 'u_action_num_ful_max_'+str(gap),
                               'type_1': 'u_browse_num_ful_max_'+str(gap),
                               'type_2': 'u_add_num_ful_max_'+str(gap),
                               'type_3': 'u_del_num_ful_max_'+str(gap),
                               'type_4': 'u_buy_num_ful_max_'+str(gap),
                               'type_5': 'u_follow_num_ful_max_'+str(gap),
                               'type_6': 'u_click_num_ful_max_'+str(gap)}, inplace=True)
    u_feature = pd.merge(u_feature, action_avg, on='user_id', how='left')
    u_feature = pd.merge(u_feature, action_median, on='user_id', how='left')
    u_feature = pd.merge(u_feature, action_max, on='user_id', how='left').fillna(0)
    
    # 行为天数
    if gap != str(1):
        action_ful['date'] = action_ful.time.apply(lambda x:x.date())
        action_days = action_ful[['user_id', 'date']]
        action_days = action_days.drop_duplicates()
        action_days = action_days.groupby('user_id').count().reset_index()
        action_days.rename(columns={'date': 'u_action_ful_days_'+str(gap)}, inplace=True)
        u_feature = pd.merge(u_feature, action_days, on='user_id', how='left').fillna(0)  
        
    # 时间特征
    action_days = action_ful[['user_id', 'time']]
    action_start = action_days.groupby('user_id').min().reset_index()
    action_start.rename(columns={'time': 'start'}, inplace=True)
    action_end = action_days.groupby('user_id').max().reset_index()
    action_end.rename(columns={'time': 'end'}, inplace=True)
    action_duration = pd.merge(action_start, action_end, on='user_id')
    action_duration['u_action_full_attentionday_'+str(gap)] = sep_day - action_duration['start']
    action_duration['u_action_full_attentionday_'+str(gap)] = action_duration['u_action_full_attentionday_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration['u_action_full_duration_'+str(gap)] = action_duration['end'] - action_duration['start']
    action_duration['u_action_full_duration_'+str(gap)] = action_duration['u_action_full_duration_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration['u_silence_full_duration_'+str(gap)] = sep_day - action_duration['end']
    action_duration['u_silence_full_duration_'+str(gap)]= action_duration['u_silence_full_duration_'+str(gap)].map(lambda x: x.days*24+x.seconds/3600)
    action_duration = action_duration[['user_id','u_action_full_attentionday_'+str(gap) ,'u_action_full_duration_'+str(gap),'u_silence_full_duration_'+str(gap)]]
    u_feature = pd.merge(u_feature, action_duration, on='user_id', how='left').fillna(0)    
      
    # 平均行为  
    u_feature['u_action_num_per_ful_day_'+str(gap)] = u_feature['u_action_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_browse_num_per_ful_day_'+str(gap)] = u_feature['u_browse_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_add_num_per_ful_day_'+str(gap)] = u_feature['u_add_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_del_num_per_ful_day_'+str(gap)] = u_feature['u_del_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_buy_num_per_ful_day_'+str(gap)] = u_feature['u_buy_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_follow_num_per_ful_day_'+str(gap)] = u_feature['u_follow_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    u_feature['u_click_num_per_ful_day_'+str(gap)] = u_feature['u_click_num_ful_'+str(gap)]/(u_feature['u_action_full_duration_'+str(gap)]+0.001)
    
    # 点击模块 (14, 21, 28, 110, 210, 216, 217)
    click_history = action_ful[action_ful['type']==6].fillna(-1)
    click_history = click_history[['user_id', 'model_id']]
    click_history['u_click14_ful_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==14))
    click_history['u_click21_ful_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==21))
    click_history['u_click28_ful_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==28))
    click_history['u_click110_ful_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==110))
    click_history['u_click210_ful_'+str(gap)] = click_history['model_id'].map(lambda x: int(x==210))
    click_history = click_history.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
    u_feature = pd.merge(u_feature, click_history, on='user_id', how='left')
 
    # 商品特征
    u_feature['u_click14_ful/click_sum_ful_'+str(gap)] = u_feature['u_click14_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_click21_ful/click_sum_ful_'+str(gap)] = u_feature['u_click21_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_click28_ful/click_sum_ful_'+str(gap)] = u_feature['u_click28_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_click110_ful/click_sum_ful_'+str(gap)] = u_feature['u_click110_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_click210_ful/click_sum_ful_'+str(gap)] = u_feature['u_click210_ful_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
    
    user_sku = action_dummy.drop('user_id', axis=1).groupby('sku_id').sum().reset_index()
    user_sku = pd.merge(action_dummy[['user_id', 'sku_id']].drop_duplicates(), user_sku, on='sku_id', how='left')
    user_sku = user_sku.drop('sku_id', axis=1)
    user_sku_avg = user_sku.groupby('user_id').mean().reset_index()
    user_sku_avg.rename(columns={'type_0': 'u_sku_action_num_ful_avg_'+str(gap),
                                 'type_1': 'u_sku_browse_num_ful_avg_'+str(gap),
                                 'type_2': 'u_sku_add_num_ful_avg_'+str(gap),
                                 'type_3': 'u_sku_del_num_ful_avg_'+str(gap),
                                 'type_4': 'u_sku_buy_num_ful_avg_'+str(gap),
                                 'type_5': 'u_sku_follow_num_ful_avg_'+str(gap),
                                 'type_6': 'u_sku_click_num_ful_avg_'+str(gap)}, inplace=True)
    user_sku_min = user_sku.groupby('user_id').min().reset_index()
    user_sku_min.rename(columns={'type_0': 'u_sku_action_num_ful_min_'+str(gap),
                                 'type_1': 'u_sku_browse_num_ful_min_'+str(gap),
                                 'type_2': 'u_sku_add_num_ful_min_'+str(gap),
                                 'type_3': 'u_sku_del_num_ful_min_'+str(gap),
                                 'type_4': 'u_sku_buy_num_ful_min_'+str(gap),
                                 'type_5': 'u_sku_follow_num_ful_min_'+str(gap),
                                 'type_6': 'u_sku_click_num_ful_min_'+str(gap)}, inplace=True)
    u_feature = pd.merge(u_feature, user_sku_avg, on='user_id', how='left')
    u_feature = pd.merge(u_feature, user_sku_min, on='user_id', how='left').fillna(0)   
        
    # 子集/全集
    u_feature['u_action_num/ful_'+str(gap)] = u_feature['u_action_num_'+str(gap)]/(u_feature['u_action_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_browse_num/ful_'+str(gap)] = u_feature['u_browse_num_'+str(gap)]/(u_feature['u_browse_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_add_num/ful_'+str(gap)] = u_feature['u_add_num_'+str(gap)]/(u_feature['u_add_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_del_num/ful_'+str(gap)] = u_feature['u_del_num_'+str(gap)]/(u_feature['u_del_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_buy_num/ful_'+str(gap)] = u_feature['u_buy_num_'+str(gap)]/(u_feature['u_buy_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_follow_num/ful_'+str(gap)] = u_feature['u_follow_num_'+str(gap)]/(u_feature['u_follow_num_ful_'+str(gap)]+0.001)*100
    u_feature['u_click_num/ful_'+str(gap)] = u_feature['u_click_num_'+str(gap)]/(u_feature['u_click_num_ful_'+str(gap)]+0.001)*100
  

  
    u_feature = pd.merge(user[['user_id', 'user_lv_cd', 'reg_duration', 'reg_duration_cate']], u_feature, on='user_id', how='left').fillna(0)
    u_feature['u_lv/reg'] = u_feature['user_lv_cd']/(u_feature['reg_duration']+0.001)*100
    u_feature['u_lv/reg_cate'] = u_feature['user_lv_cd']/(u_feature['reg_duration_cate']+0.001)
    
    return u_feature
#%%
ful_action = pd.read_csv('../../data/JData_Action.csv', parse_dates=[2], infer_datetime_format=True)
sel_action = pd.read_csv('../../data/JData_Selected_action.csv', parse_dates=[2, 7], infer_datetime_format=True)        
user = pd.read_csv('../../data/JData_Coded_user.csv', parse_dates=[4])


get_features(ful_action, sel_action, user)





