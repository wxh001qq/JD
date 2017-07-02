
import pandas as pd
import numpy as np
import datetime
import math

path = '../data'

def features(ful_action, sel_action, user):
# *** set date ***
    sep_day = datetime.datetime(2016, 4, 6)
    predict_period = datetime.timedelta(5, 0)
    day = datetime.timedelta(1, 0)
    start_day = datetime.datetime(2016,1,31)

    print('==========> 获取用户特征...')
        
# *** feature extracting ***
    for k in range(3):
        print('%d / 3' %(k+1))
        duration = sep_day - start_day
        duration = duration.days
# *** generate action subset ***
        user = user.drop_duplicates('user_id')
    
        action_history = sel_action[sel_action['time']<sep_day]
        action_30D = sel_action[(sel_action['time']>=sep_day-30*day) & (sel_action['time']<sep_day)]
        action_15D = sel_action[(sel_action['time']>=sep_day-15*day) & (sel_action['time']<sep_day)]
        action_5D = sel_action[(sel_action['time']>=sep_day-5*day) & (sel_action['time']<sep_day)]
        action_3D = sel_action[(sel_action['time']>=sep_day-3*day) & (sel_action['time']<sep_day)]
        action_1D = sel_action[(sel_action['time']>=sep_day-1*day) & (sel_action['time']<sep_day)]
        
        ful_action_history = ful_action[ful_action['time']<sep_day]
        ful_action_5D = ful_action[(ful_action['time']>=sep_day-5*day) & (ful_action['time']<sep_day)]
        ful_action_30D = ful_action[(ful_action['time']>=sep_day-30*day) & (ful_action['time']<sep_day)]
        
        feature_i = user_features(user, ful_action_history, ful_action_30D, ful_action_5D, action_history, action_30D, action_5D, action_3D, action_1D, sep_day)
    
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
                feature_i.to_csv(path + '/feature03/feature_train.csv', index=False)
            else:
                feature_i.drop('label', axis=1).to_csv(path + '/feature03/feature_val.csv', index=False)
        else:
            feature_i.to_csv(path + '/feature03/feature_test.csv', index=False)
        
        sep_day += predict_period
    print('==========> 完成')


# In[16]:

def user_features(user, action_ful_history, action_ful_30D, action_ful_5D, action_history, action_30D, action_5D, action_3D, action_1D, sep_day):
    
    print('=====> 提取特征...')
    
    # ========================================
    #    用户历史行为  
    # ========================================
    print('-- History')
    # 6种行为特征
    action_dummy = pd.get_dummies(action_history['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_history[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature_history = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_history.rename(columns={'type_0': 'u_action_num_history',
                                      'type_1': 'u_browse_num_history',
                                      'type_2': 'u_add_num_history',
                                      'type_3': 'u_del_num_history',
                                      'type_4': 'u_buy_num_history',
                                      'type_5': 'u_follow_num_history',
                                      'type_6': 'u_click_num_history'}, inplace=True)
    
    # 比值
    u_feature_history['u_buy/browse_num_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_browse_num_history']+0.001)*100
    u_feature_history['u_buy/add_num_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_add_num_history']+0.001)*100
    u_feature_history['u_buy/click_num_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history['u_buy/follow_num_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_follow_num_history']+0.001)*100
    u_feature_history['u_del/add_num_history'] = u_feature_history['u_del_num_history']/(u_feature_history['u_add_num_history']+0.001)*100
    
    # 用户对商品行为特征
    action_sku = action_dummy.groupby(['user_id', 'sku_id']).sum().reset_index()
    action_sku = action_sku.drop('sku_id', axis=1)
    action_avg = action_sku.groupby('user_id').mean().reset_index()
    action_avg.rename(columns={'type_0': 'u_action_num_avg_history',
                               'type_1': 'u_browse_num_avg_history',
                               'type_2': 'u_add_num_avg_history',
                               'type_3': 'u_del_num_avg_history',
                               'type_4': 'u_buy_num_avg_history',
                               'type_5': 'u_follow_num_avg_history',
                               'type_6': 'u_click_num_avg_history'}, inplace=True)
    action_max = action_sku.groupby('user_id').max().reset_index()
    action_max.rename(columns={'type_0': 'u_action_num_max_history',
                               'type_1': 'u_browse_num_max_history',
                               'type_2': 'u_add_num_max_history',
                               'type_3': 'u_del_num_max_history',
                               'type_4': 'u_buy_num_max_history',
                               'type_5': 'u_follow_num_max_history',
                               'type_6': 'u_click_num_max_history'}, inplace=True)
    action_max = action_max.drop(['u_buy_num_max_history', 'u_del_num_max_history'], axis=1)
    u_feature_history = pd.merge(u_feature_history, action_avg, on='user_id', how='left')
    u_feature_history = pd.merge(u_feature_history, action_max, on='user_id', how='left').fillna(0)
    
    # 活跃天数
    action_days = action_history[['user_id', 'date']]
    action_days = action_days.drop_duplicates()
    action_days = action_days.groupby('user_id').count().reset_index()
    action_days.rename(columns={'date': 'u_action_days_history'}, inplace=True)
    u_feature_history = pd.merge(u_feature_history, action_days, on='user_id', how='left').fillna(0)
    
    # 时间特征
    action_days = action_history[['user_id', 'time']]
    action_start = action_days.groupby('user_id').min().reset_index()
    action_start.rename(columns={'time': 'start'}, inplace=True)
    action_end = action_days.groupby('user_id').max().reset_index()
    action_end.rename(columns={'time': 'end'}, inplace=True)
    action_duration = pd.merge(action_start, action_end, on='user_id')
    action_duration['u_action_duration_history'] = action_duration['end'] - action_duration['start']
    action_duration['u_action_duration_history'] = action_duration['u_action_duration_history'].map(lambda x: x.days*24+x.seconds/3600)
    action_duration = action_duration[['user_id', 'u_action_duration_history']]
    u_feature_history = pd.merge(u_feature_history, action_duration, on='user_id', how='left').fillna(0)
    
    # 行为/时间
    u_feature_history['u_action_num_per_day_history'] = u_feature_history['u_action_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_browse_num_per_day_history'] = u_feature_history['u_browse_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_add_num_per_day_history'] = u_feature_history['u_add_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_del_num_per_day_history'] = u_feature_history['u_del_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_buy_num_per_day_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_follow_num_per_day_history'] = u_feature_history['u_follow_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    u_feature_history['u_click_num_per_day_history'] = u_feature_history['u_click_num_history']/(u_feature_history['u_action_duration_history']+0.001)
    
    # 点击模块 (14, 21, 28, 110, 210, 216, 217)
    click_history = action_history[action_history['type']==6].fillna(-1)
    click_history = click_history[['user_id', 'model_id']]
    click_history['u_click14_history'] = click_history['model_id'].map(lambda x: int(x==14))
    click_history['u_click21_history'] = click_history['model_id'].map(lambda x: int(x==21))
    click_history['u_click28_history'] = click_history['model_id'].map(lambda x: int(x==28))
    click_history['u_click110_history'] = click_history['model_id'].map(lambda x: int(x==110))
    click_history['u_click210_history'] = click_history['model_id'].map(lambda x: int(x==210))
    click_history = click_history.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
    u_feature_history = pd.merge(u_feature_history, click_history, on='user_id', how='left')
    
    # 点击模块/点击
    u_feature_history['u_click14/click_sum_history'] = u_feature_history['u_click14_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history['u_click21/click_sum_history'] = u_feature_history['u_click21_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history['u_click28/click_sum_history'] = u_feature_history['u_click28_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history['u_click110/click_sum_history'] = u_feature_history['u_click110_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history['u_click210/click_sum_history'] = u_feature_history['u_click210_history']/(u_feature_history['u_click_num_history']+0.001)*100
    u_feature_history = u_feature_history.drop(['u_click110_history', 'u_click28_history'], axis=1)
    
    # 商品特征
    user_sku = action_dummy.drop('user_id', axis=1).groupby('sku_id').sum().reset_index()
    user_sku = pd.merge(action_dummy[['user_id', 'sku_id']].drop_duplicates(), user_sku, on='sku_id', how='left')
    user_sku = user_sku.drop('sku_id', axis=1)
    user_sku_avg = user_sku.groupby('user_id').mean().reset_index()
    user_sku_avg.rename(columns={'type_0': 'u_sku_action_num_avg_history',
                                 'type_1': 'u_sku_browse_num_avg_history',
                                 'type_2': 'u_sku_add_num_avg_history',
                                 'type_3': 'u_sku_del_num_avg_history',
                                 'type_4': 'u_sku_buy_num_avg_history',
                                 'type_5': 'u_sku_follow_num_avg_history',
                                 'type_6': 'u_sku_click_num_avg_history'}, inplace=True)
    user_sku_min = user_sku.groupby('user_id').min().reset_index()
    user_sku_min.rename(columns={'type_0': 'u_sku_action_num_min_history',
                                 'type_1': 'u_sku_browse_num_min_history',
                                 'type_2': 'u_sku_add_num_min_history',
                                 'type_3': 'u_sku_del_num_min_history',
                                 'type_4': 'u_sku_buy_num_min_history',
                                 'type_5': 'u_sku_follow_num_min_history',
                                 'type_6': 'u_sku_click_num_min_history'}, inplace=True)
    u_feature_history = pd.merge(u_feature_history, user_sku_avg, on='user_id', how='left')
    u_feature_history = pd.merge(u_feature_history, user_sku_min, on='user_id', how='left').fillna(0)
    
    # 全集行为特征
    action_dummy = pd.get_dummies(action_ful_history['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_ful_history[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature_ful_history = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_ful_history.rename(columns={'type_0': 'u_action_num_ful_history',
                                          'type_1': 'u_browse_num_ful_history',
                                          'type_2': 'u_add_num_ful_history',
                                          'type_3': 'u_del_num_ful_history',
                                          'type_4': 'u_buy_num_ful_history',
                                          'type_5': 'u_follow_num_ful_history',
                                          'type_6': 'u_click_num_ful_history'}, inplace=True)
    u_feature_history = pd.merge(u_feature_history, u_feature_ful_history, on='user_id', how='left')
    
    # 子集/全集
    u_feature_history['u_action_num/ful_history'] = u_feature_history['u_action_num_history']/(u_feature_history['u_action_num_ful_history']+0.001)*100
    u_feature_history['u_browse_num/ful_history'] = u_feature_history['u_browse_num_history']/(u_feature_history['u_browse_num_ful_history']+0.001)*100
    u_feature_history['u_add_num/ful_history'] = u_feature_history['u_add_num_history']/(u_feature_history['u_add_num_ful_history']+0.001)*100
    u_feature_history['u_del_num/ful_history'] = u_feature_history['u_del_num_history']/(u_feature_history['u_del_num_ful_history']+0.001)*100
    u_feature_history['u_buy_num/ful_history'] = u_feature_history['u_buy_num_history']/(u_feature_history['u_buy_num_ful_history']+0.001)*100
    u_feature_history['u_follow_num/ful_history'] = u_feature_history['u_follow_num_history']/(u_feature_history['u_follow_num_ful_history']+0.001)*100
    u_feature_history['u_click_num/ful_history'] = u_feature_history['u_click_num_history']/(u_feature_history['u_click_num_ful_history']+0.001)*100
    u_feature_history = u_feature_history.drop(['u_action_num_ful_history', 'u_browse_num_ful_history', 'u_add_num_ful_history', 'u_del_num_ful_history',
                                                'u_buy_num_ful_history', 'u_follow_num_ful_history', 'u_click_num_ful_history'], axis=1)
    
    
    # =======================================
    #    用户30天行为特征
    # =======================================
    print('-- 30 days')
    action_dummy = pd.get_dummies(action_30D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_30D[['user_id', 'sku_id']], action_dummy], axis=1)
    # 子集行为特征
    u_feature_30D = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_30D.rename(columns={'type_0': 'u_action_num_30D',
                                  'type_1': 'u_browse_num_30D',
                                  'type_2': 'u_add_num_30D',
                                  'type_3': 'u_del_num_30D',
                                  'type_4': 'u_buy_num_30D',
                                  'type_5': 'u_follow_num_30D',
                                  'type_6': 'u_click_num_30D'}, inplace=True)
    # 全集行为特征
    action_dummy = pd.get_dummies(action_ful_30D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_ful_30D[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature_ful_30D = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_ful_30D.rename(columns={'type_0': 'u_action_num_ful_30D',
                                      'type_1': 'u_browse_num_ful_30D',
                                      'type_2': 'u_add_num_ful_30D',
                                      'type_3': 'u_del_num_ful_30D',
                                      'type_4': 'u_buy_num_ful_30D',
                                      'type_5': 'u_follow_num_ful_30D',
                                      'type_6': 'u_click_num_ful_30D'}, inplace=True)
    u_feature_30D = pd.merge(u_feature_30D, u_feature_ful_30D, on='user_id', how='left')
    
    # 子集/全集
    u_feature_30D['u_action_num/ful_30D'] = u_feature_30D['u_action_num_30D']/(u_feature_30D['u_action_num_ful_30D']+0.001)*100
    u_feature_30D['u_browse_num/ful_30D'] = u_feature_30D['u_browse_num_30D']/(u_feature_30D['u_browse_num_ful_30D']+0.001)*100
    u_feature_30D['u_add_num/ful_30D'] = u_feature_30D['u_add_num_30D']/(u_feature_30D['u_add_num_ful_30D']+0.001)*100
    u_feature_30D['u_del_num/ful_30D'] = u_feature_30D['u_del_num_30D']/(u_feature_30D['u_del_num_ful_30D']+0.001)*100
    u_feature_30D['u_buy_num/ful_30D'] = u_feature_30D['u_buy_num_30D']/(u_feature_30D['u_buy_num_ful_30D']+0.001)*100
    u_feature_30D['u_follow_num/ful_30D'] = u_feature_30D['u_follow_num_30D']/(u_feature_30D['u_follow_num_ful_30D']+0.001)*100
    u_feature_30D['u_click_num/ful_30D'] = u_feature_30D['u_click_num_30D']/(u_feature_30D['u_click_num_ful_30D']+0.001)*100
    
    # ========================================
    #     用户5天行为特征
    # ========================================
    print('-- 5 days')
    action_dummy = pd.get_dummies(action_5D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_5D[['user_id', 'sku_id']], action_dummy], axis=1)
    # 子集行为特征
    u_feature_5D = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_5D.rename(columns={'type_0': 'u_action_num_5D',
                                 'type_1': 'u_browse_num_5D',
                                 'type_2': 'u_add_num_5D',
                                 'type_3': 'u_del_num_5D',
                                 'type_4': 'u_buy_num_5D',
                                 'type_5': 'u_follow_num_5D',
                                 'type_6': 'u_click_num_5D'}, inplace=True)
    
    # 用户对商品行为特征
    action_sku = action_dummy.groupby(['user_id', 'sku_id']).sum().reset_index()
    action_sku = action_sku.drop('sku_id', axis=1)
    action_avg = action_sku.groupby('user_id').mean().reset_index()
    action_avg.rename(columns={'type_0': 'u_action_num_avg_5D',
                               'type_1': 'u_browse_num_avg_5D',
                               'type_2': 'u_add_num_avg_5D',
                               'type_3': 'u_del_num_avg_5D',
                               'type_4': 'u_buy_num_avg_5D',
                               'type_5': 'u_follow_num_avg_5D',
                               'type_6': 'u_click_num_avg_5D'}, inplace=True)
    action_max = action_sku.groupby('user_id').max().reset_index()
    action_max.rename(columns={'type_0': 'u_action_num_max_5D',
                               'type_1': 'u_browse_num_max_5D',
                               'type_2': 'u_add_num_max_5D',
                               'type_3': 'u_del_num_max_5D',
                               'type_4': 'u_buy_num_max_5D',
                               'type_5': 'u_follow_num_max_5D',
                               'type_6': 'u_click_num_max_5D'}, inplace=True)
    u_feature_5D = pd.merge(u_feature_5D, action_avg, on='user_id', how='left')
    u_feature_5D = pd.merge(u_feature_5D, action_max, on='user_id', how='left').fillna(0)
    
    # 时间特征
    action_days = action_5D[['user_id', 'time']]
    action_start = action_days.groupby('user_id').min().reset_index()
    action_start.rename(columns={'time': 'start'}, inplace=True)
    action_end = action_days.groupby('user_id').max().reset_index()
    action_end.rename(columns={'time': 'end'}, inplace=True)
    action_duration = pd.merge(action_start, action_end, on='user_id')
    action_duration['u_action_duration_5D'] = action_duration['end'] - action_duration['start']
    action_duration['u_action_duration_5D'] = action_duration['u_action_duration_5D'].map(lambda x: x.days*24+x.seconds/3600)
    action_duration['u_silence_duration_5D'] = sep_day - action_duration['end']
    action_duration['u_silence_duration_5D']= action_duration['u_silence_duration_5D'].map(lambda x: x.days*24+x.seconds/3600)
    action_duration = action_duration[['user_id', 'u_action_duration_5D', 'u_silence_duration_5D']]
    u_feature_5D = pd.merge(u_feature_5D, action_duration, on='user_id', how='left').fillna(0)
    
    # 点击模块 (14, 21, 28, 110, 210, 216, 217)
    click_5D = action_5D[action_5D['type']==6].fillna(-1)
    click_5D = click_5D[['user_id', 'model_id']]
    click_5D['u_click14_5D'] = click_5D['model_id'].map(lambda x: int(x==14))
    click_5D['u_click21_5D'] = click_5D['model_id'].map(lambda x: int(x==21))
    click_5D['u_click28_5D'] = click_5D['model_id'].map(lambda x: int(x==28))
    click_5D['u_click110_5D'] = click_5D['model_id'].map(lambda x: int(x==110))
    click_5D['u_click210_5D'] = click_5D['model_id'].map(lambda x: int(x==210))
    click_5D = click_5D.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
    u_feature_5D = pd.merge(u_feature_5D, click_5D, on='user_id', how='left')
    
    # 点击模块/点击
    u_feature_5D['u_click14/click_sum_5D'] = u_feature_5D['u_click14_5D']/(u_feature_5D['u_click_num_5D']+0.001)*100
    u_feature_5D['u_click21/click_sum_5D'] = u_feature_5D['u_click21_5D']/(u_feature_5D['u_click_num_5D']+0.001)*100
    u_feature_5D['u_click28/click_sum_5D'] = u_feature_5D['u_click28_5D']/(u_feature_5D['u_click_num_5D']+0.001)*100
    u_feature_5D['u_click110/click_sum_5D'] = u_feature_5D['u_click110_5D']/(u_feature_5D['u_click_num_5D']+0.001)*100
    u_feature_5D['u_click210/click_sum_5D'] = u_feature_5D['u_click210_5D']/(u_feature_5D['u_click_num_5D']+0.001)*100
    u_feature_5D = u_feature_5D.drop(['u_click110_5D', 'u_click28_5D'], axis=1)
    
    # 全集行为特征
    action_dummy = pd.get_dummies(action_ful_5D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_ful_5D[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature_ful_5D = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_ful_5D.rename(columns={'type_0': 'u_action_num_ful_5D',
                                     'type_1': 'u_browse_num_ful_5D',
                                     'type_2': 'u_add_num_ful_5D',
                                     'type_3': 'u_del_num_ful_5D',
                                     'type_4': 'u_buy_num_ful_5D',
                                     'type_5': 'u_follow_num_ful_5D',
                                     'type_6': 'u_click_num_ful_5D'}, inplace=True)
    # 子集/全集
    u_feature_5D = pd.merge(u_feature_5D, u_feature_ful_5D, on='user_id', how='left')
    u_feature_5D['u_browse_num/ful_5D'] = u_feature_5D['u_browse_num_5D'] / (u_feature_5D['u_browse_num_ful_5D']+0.001)*100
    u_feature_5D['u_add_num/ful_5D'] = u_feature_5D['u_add_num_5D'] / (u_feature_5D['u_add_num_ful_5D']+0.001)*100
    u_feature_5D['u_del_num/ful_5D'] = u_feature_5D['u_del_num_5D'] / (u_feature_5D['u_del_num_ful_5D']+0.001)*100
    u_feature_5D['u_click_num/ful_5D'] = u_feature_5D['u_click_num_5D'] / (u_feature_5D['u_click_num_ful_5D']+0.001)*100
    #u_feature_5D = u_feature_5D.drop(['u_browse_num_ful_5D','u_add_num_ful_5D','u_del_num_ful_5D','u_buy_num_ful_5D','u_follow_num_ful_5D','u_click_num_ful_5D'], axis=1)
    
    # ========================================
    #     用户3天行为特征  
    # ========================================
    print('-- 3 days')
    action_dummy = pd.get_dummies(action_3D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_3D[['user_id', 'sku_id']], action_dummy], axis=1)
    u_feature_3D = action_dummy.groupby('user_id')['type_0'].sum().reset_index()
    u_feature_3D.rename(columns={'type_0': 'u_action_num_3D'}, inplace=True)
    
    # ========================================
    #     用户1天行为特征  
    # ========================================
    print('-- 1 day')
    action_dummy = pd.get_dummies(action_1D['type'], prefix='type')
    action_dummy['type_0'] = action_dummy.sum(axis=1)
    action_dummy = pd.concat([action_1D[['user_id', 'sku_id']], action_dummy], axis=1)
    
    u_feature_1D = action_dummy.drop('sku_id', axis=1).groupby('user_id').sum().reset_index()
    u_feature_1D.rename(columns={'type_0': 'u_action_num_1D',
                                 'type_1': 'u_browse_num_1D',
                                 'type_2': 'u_add_num_1D',
                                 'type_3': 'u_del_num_1D',
                                 'type_4': 'u_buy_num_1D',
                                 'type_5': 'u_follow_num_1D',
                                 'type_6': 'u_click_num_1D'}, inplace=True)
    # ========================================
    #          特征融合
    # ========================================
    print('-- Merging')
    u_feature = pd.merge(user[['user_id', 'user_lv_cd', 'reg_duration', 'reg_duration_cate']], u_feature_history, on='user_id', how='left')
    u_feature['u_lv/reg'] = u_feature['user_lv_cd']/(u_feature['reg_duration']+0.001)*100
    u_feature['u_lv/reg_cate'] = u_feature['user_lv_cd']/(u_feature['reg_duration_cate']+0.001)
    u_feature = pd.merge(u_feature, u_feature_30D, on='user_id', how='left')
    u_feature = pd.merge(u_feature, u_feature_5D, on='user_id', how='left')
    u_feature['user_action_num_5D/history'] = u_feature['u_action_num_5D']/(u_feature['u_action_num_history']+0.001)
    u_feature = pd.merge(u_feature, u_feature_3D, on='user_id', how='left')
    u_feature = pd.merge(u_feature, u_feature_1D, on='user_id', how='left').fillna(0)
    
    u_feature['u_action_num_difference_1D'] = u_feature['u_action_num_1D']-u_feature['u_action_num_per_day_history']
    u_feature['u_browse_num_difference_1D'] = u_feature['u_browse_num_1D'] - u_feature['u_browse_num_per_day_history']
    u_feature['u_add_num_difference_1D'] = u_feature['u_add_num_1D'] - u_feature['u_add_num_per_day_history']
    u_feature['u_del_num_difference_1D'] = u_feature['u_del_num_1D'] - u_feature['u_del_num_per_day_history']
    u_feature['u_buy_num_difference_1D'] = u_feature['u_buy_num_1D'] - u_feature['u_buy_num_per_day_history']
    u_feature['u_follow_num_difference_1D'] = u_feature['u_follow_num_1D'] - u_feature['u_follow_num_per_day_history']
    u_feature['u_click_num_difference_1D'] = u_feature['u_click_num_1D'] - u_feature['u_click_num_per_day_history']
    
    print('=====> 完成!')
    return u_feature


ful_action = pd.read_csv('../data/JData_Action.csv', parse_dates=[2])
sel_action = pd.read_csv('../data/JData_subset_action.csv', parse_dates=[2])
user = pd.read_csv('../data/JData_modified_user.csv')

features(ful_action, sel_action, user)




