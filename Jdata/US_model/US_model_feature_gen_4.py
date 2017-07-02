#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#from basic_feat0518 import *

# 所有行为的总和
def get_action_feat(start_date, end_date):
    dump_path = './cache/action_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        actions.to_csv(dump_path, index=False)
    return actions

# 行为按时间衰减
def get_accumulate_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions.columns = ['user_id', 'sku_id', 'time', 'model_id', 'type',
                           'cate', 'brand', 'action_1', 'action_2', 'action_3',
                           'action_4', 'action_5', 'action_6']
        # 近期行为按时间衰减
        actions['weights'] = actions['time'].map(
            lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights_1'] = actions['weights'].map(lambda x: 0.01938 * (x.days + 1) ** (-0.73563))
        actions['weights_2'] = actions['weights'].map(lambda x: 0.054465 * (x.days + 1) ** (-0.9035))
        actions['weights_3'] = actions['weights'].map(lambda x: 0.012186 * (x.days + 1) ** (-0.6953))
        actions['weights_4'] = actions['weights'].map(lambda x: 0.01357 * (x.days + 1) ** (-0.467889))
        actions['weights_5'] = actions['weights'].map(lambda x: 0.05234 * (x.days + 1) ** (-0.80797))
        actions['weights_6'] = actions['weights'].map(lambda x: 0.019337 * (x.days + 1) ** (-0.7341655))
        actions['action_1'] = actions['action_1'] * actions['weights_1']
        actions['action_2'] = actions['action_2'] * actions['weights_2']
        actions['action_3'] = actions['action_3'] * actions['weights_3']
        actions['action_4'] = actions['action_4'] * actions['weights_4']
        actions['action_5'] = actions['action_5'] * actions['weights_5']
        actions['action_6'] = actions['action_6'] * actions['weights_6']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['weights']
        del actions['weights_1']
        del actions['weights_2']
        del actions['weights_3']
        del actions['weights_4']
        del actions['weights_5']
        del actions['weights_6']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        actions.to_csv(dump_path, index=False)
    return actions

# U-B对浏览次数/用户总浏览次数
# U_B对行为1，2，4，5进行 浏览次数/用户总浏览次数（或者物品的浏览次数）
def get_action_U_P_feat1(start_date, end_date):
    dump_path = './cache/U_B_feat1_eight_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        actions.columns = ['user_id', 'sku_id'] + ['us_feat1_' + str(i) for i in range(1, actions.shape[1] - 1)]
        return actions
    else:
        temp = None
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        for i in (1, 2, 4, 5):
            actions = df[df['type'] == i]
            # del actions['type']
            action1 = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
            action1.columns = ['user_id', 'sku_id', 'visit']
            action2 = actions.groupby('user_id', as_index=False).count()
            del action2['type']
            action2.columns = ['user_id', 'user_visits']

            action3 = actions.groupby('sku_id', as_index=False).count()
            del action3['type']
            action3.columns = ['sku_id', 'sku_visits']

            actions = pd.merge(action1, action2, how='left', on='user_id')
            actions = pd.merge(actions, action3, how='left', on='sku_id')
            actions['visit_rate_user'] = actions['visit'] / actions['user_visits']
            actions['visit_rate_sku'] = actions['visit'] / actions['sku_visits']
            del actions['visit']
            del actions['user_visits']
            del actions['sku_visits']
            if temp is None:
                temp = actions
            else:
                temp = pd.merge(temp, actions, how="outer", on=['user_id', 'sku_id'])
        temp.to_csv(dump_path, index=False)
        return temp


# 用户关注或加入购物车，但是不购买，且加入购物车或者关注小于10天
def get_action_U_P_feat2(start_date, end_date):
    dump_path = './cache/U_B_feat2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)
        # print df[df['type']==2].shape#296891
        # print df[df['type']==6].shape#16627100
        df1 = df[(df['type'] == 2) | (df['type'] == 6)][['user_id', 'sku_id', 'time']]
        df2 = df[df['type'] == 4][['user_id', 'sku_id']]
        df2['label'] = 0
        actions = pd.merge(df1, df2, on=['user_id', 'sku_id'], how='left')
        actions = actions.fillna(1)
        actions['time'] = (actions['time'].map(
            lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))).dt.days
        #         actions[actions['time']>10].loc['label']=0
        actions.loc[actions['time'] > 10, 'label'] = 0
        del actions['time']
        actions.columns = ['user_id', 'sku_id', 'notbuy']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat2_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


def user_product_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del actions['type']
    user_sku = actions[['user_id', 'sku_id']]
    del actions['sku_id']
    del actions['user_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([user_sku, actions], axis=1)
    return actions


# print user_product_top_k_0_1('2016-03-10','2016-04-11')
# 最近K天行为0/1提取
def get_action_U_P_feat3(k, start_date, end_date):
    dump_path = './cache/U_P_feat3_%s_%s_%s.csv' % (k, start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=k)
        start_days = start_days.strftime('%Y-%m-%d')
        actions = user_product_top_k_0_1(start_days, end_date)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat3_' + str(k) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


# 获取货物最近一次行为的时间距离当前时间的差距
def get_action_U_P_feat4(start_date, end_date):
    dump_path = './cache/U_P_feat4_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['user_id', 'sku_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['user_id', 'sku_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat4_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 获取最后一次行为的次数并且进行归一化
def get_action_U_P_feat5(start_date, end_date):
    dump_path = './cache/U_P_feat5_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['user_id', 'sku_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["user_id", "sku_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat5_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 获取人物和商品该层级最后一层的各种行为的统计数量
def get_action_U_P_feat6(start_date, end_date, n):
    dump_path = './cache/U_P_feat6_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat6_' + str(n) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


# 品牌层级天数
def get_action_U_P_feat7(start_date, end_date):
    dump_path = './cache/U_P_feat7_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['user_id', 'sku_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        columns = []
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['user_id', 'sku_id', 'level%s' % i, 'time']]
                df = df.groupby(['user_id', 'sku_id', 'level%s' % i]).count()
                df = df.unstack()
                df = df.reset_index()
                df.columns = ['user_id', 'sku_id'] + list(range(df.shape[1] - 2))
                print (df.head())
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on=['user_id', 'sku_id'], how='left')
        user_sku = result[['user_id', 'sku_id']]
        del result['sku_id']
        del result['user_id']
        actions = result.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions.columns = range(actions.shape[1])
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat7_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 用户和商品过去购六种行为频数
def get_action_U_P_feat8(start_date, end_date):
    dump_path = './cache/U_P_feat8_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type', 'time']]
        actions = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).count()
        time_min = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'sku_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'sku_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['user_id', 'sku_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'sku_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat8_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# #层级天数  ,一共有几天产生了购买行为
def get_action_U_P_feat9(start_date, end_date, n):
    dump_path = './cache/U_P_feat9_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['user_id', 'sku_id', 'type', 'time'], keep='first')

        actions = df.groupby(['user_id', 'sku_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat9_' + str(n) + '_' + str(i) for i in
                                               range(1, actions.shape[1] - 1)]
    return actions


def get_action_U_P_feat14(start_date, end_date):
    dump_path = './cache/U_P_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'time', 'type']]
        df = df[df['type'] == 4][['user_id', 'sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['user_id', 'sku_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['user_id', 'sku_id']]
        del actions['user_id']
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat14_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions


# 用户和品牌交叉
def get_action_U_P_feat16(start_date, end_date):
    dump_path = './cache/U_P_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['sku_id']
        action2.columns = ['user_id', 'user_cnt']

        action3 = actions.groupby('sku_id', as_index=False).count()
        del action3['user_id']
        action3.columns = ['sku_id', 'sku_cnt']

        actions = pd.merge(action1, action2, how='left', on='user_id')
        actions = pd.merge(actions, action3, how='left', on='sku_id')

        actions['user_cnt'] = actions['cnt'] / actions['user_cnt']
        actions['sku_cnt'] = actions['cnt'] / actions['sku_cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id', 'sku_id'] + ['us_feat16_' + str(i) for i in range(1, actions.shape[1] - 1)]
    return actions



# 点击模块
def get_action_U_P_feat_0509_feat_24(start_date, end_date, n):
    dump_path = './cache/get_action_U_P_feat_0509_feat_24_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        actions = get_actions(start_days, end_date)
        actions = actions[actions['type'] == 6][['user_id','sku_id','model_id']]

        actions_click_sum = actions[['user_id','sku_id', 'model_id']].groupby(['user_id','sku_id']).count().reset_index()
        actions_click_sum.columns = ['user_id','sku_id', str(n) + 'click_sum_all']
        actions[str(n) + 'p_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n) + 'p_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n) + 'p_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n) + 'p_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n) + 'p_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby(['user_id','sku_id']).sum().reset_index().drop('model_id', axis=1)

        actions = pd.merge(actions, actions_click_sum, how='left', on=['user_id','sku_id'])

        actions[str(n) + 'p_click14/click_sum_history'] = actions[str(n) + 'p_click14_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click21/click_sum_history'] = actions[str(n) + 'p_click21_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click28/click_sum_history'] = actions[str(n) + 'p_click28_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click110/click_sum_history'] = actions[str(n) + 'p_click110_history'] / actions[
            str(n) + 'click_sum_all']
        actions[str(n) + 'p_click210/click_sum_history'] = actions[str(n) + 'p_click210_history'] / actions[
            str(n) + 'click_sum_all']

        sku_id = actions[['user_id','sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id','sku_id'] + ['p0509_feat_24_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1]-1)]
    return actions












def sku_tongji_info():
    dump_1 = './cache/a_tongji/brandsales.csv'
    dump_2 = './cache/a_tongji/productsales.csv'
    actions_1 = pd.read_csv(dump_1)[['sales', 'brand', 'cate']]
    actions_2 = pd.read_csv(dump_2)
    action = pd.merge([actions_2, actions_1], on=['brand', 'cate'], how='left')
    return action

