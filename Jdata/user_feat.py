#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time

from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
user_path = "./data/JData_User.csv"
product_path = "./data/JData_Product.csv"


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


# 用户的基本信息
def get_basic_user_feat():
    dump_path = './cache/basic_user.csv'
    if os.path.exists(dump_path):
        user = pd.read_csv(dump_path)
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        user.to_csv(dump_path, index=False)
    return user

# 商品的基本信息
def get_basic_product_feat():
    dump_path = './cache/basic_product.csv'
    if os.path.exists(dump_path):
        product = pd.read_csv(dump_path)
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        product.to_csv(dump_path, index=False)
    return product

def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action


def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2


def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


# 行为数据
def get_actions(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = './cache/all_action_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        actions = pd.concat([action_1, action_2])
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        actions = pd.concat([actions, action_3])  # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        actions.to_csv(dump_path, index=False)
    # actions['user_id']=actions['user_id'].astype('int')
    return actions

# 获取两个时间相差几天
def get_day_chaju(x, end_date):
    #     x=x.split(' ')[0]
    x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    return (end_date - x).days




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






# 用户的行为转化率
def get_action_user_feat1(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = './cache/user_feat_accumulate_xiugai_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        #         actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_3_ratio'] = actions['action_3'] / actions['action_2']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        #         3.购物车删除
        actions = actions[feature]
        actions.to_csv(dump_path, index=False)
    return actions


# print get_accumulate_user_feat('2016-03-10','2016-04-11')
# 用户购买前访问天数
# 用户购买/加入购物车/关注前访问天数
def get_action_user_feat2(start_date, end_date):
    dump_path = './cache/user_feat2_after_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)

    else:
        # 用户购买前访问天数
        def user_feat_2_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户加入购物车前访问天数
        def user_feat_2_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='user_id', how='left')
            actions['visit_day_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 用户关注前访问天数
        def user_feat_2_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['user_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车天数
        def user_feat_2_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.drop_duplicates(['user_id', 'time'], keep='first')
            del addtoshopping['time']
            del actions['time']
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='user_id', how='left')
            actions['addtoshopping_day_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注天数
        def user_feat_2_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.drop_duplicates(['user_id', 'time'], keep='first')
            del guanzhu['time']
            del actions['time']
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_2_1(start_date, end_date), user_feat_2_2(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_2_3(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_2_4(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_2_5(start_date, end_date), on='user_id', how='outer')
        user_id = actions['user_id']
        del actions['user_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat2_' + str(i) for i in range(1, actions.shape[1])]
    return actions




# 用户总购买品牌数
def get_action_user_feat5(start_date, end_date):
    dump_path = './cache/user_feat5_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id']]
        actions = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
        actions = actions.groupby('user_id', as_index=False).count()
        actions.columns = ['user_id', 'sku_num']
        actions['sku_num'] = actions['sku_num'].astype('float')
        actions['sku_num'] = actions['sku_num'].map(
            lambda x: (x - actions['sku_num'].min()) / (actions['sku_num'].max() - actions['sku_num'].min()))
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat5_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户平均访问间隔
def get_action_user_feat6(start_date, end_date):
    dump_path = './cache/user_feat6_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time']]
        # df['user_id']=df['user_id'].astype('int')
        df['time'] = df['time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['user_id', 'time'], keep='first')
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('user_id', as_index=False).agg(lambda x: x['time'].diff().mean())
        actions['avg_visit'] = actions['time'].dt.days
        del actions['time']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat6_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户平均六种行为的访问间隔
def get_action_user_feat6_six(start_date, end_date):
    dump_path = './cache/user_feat6_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df = df.drop_duplicates(['user_id', 'time', 'type'], keep='first')
        actions = df.groupby(['user_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat6_six_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户购买频率
def get_action_user_feat7(start_date, end_date):
    dump_path = './cache/user_feat7_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'type', 'time']]
        actions = df.groupby(['user_id', 'type'], as_index=False).count()

        time_min = df.groupby(['user_id', 'type'], as_index=False).min()
        time_max = df.groupby(['user_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['user_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['user_id', 'type'], how="left")
        actions = actions.groupby(['user_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat7_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def user_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('user_id', as_index=False).sum()
    del actions['type']
    del actions['sku_id']
    user_id = actions['user_id']
    del actions['user_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([user_id, actions], axis=1)
    return actions


# 用户最近K天行为0/1提取
def get_action_user_feat8(start_date, end_date):
    dump_path = './cache/user_feat8_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = user_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, user_top_k_0_1(start_days, end_date), how='outer', on='user_id')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat8_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取用户的重复购买率
def get_action_user_feat8_2(start_date, end_date):
    dump_path = './cache/product_feat8_2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        df = df[df['type'] == 4]  # 购买的行为
        df = df.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['user_id'], as_index=False)
        actions = grouped.count()[['user_id', 'count1']]
        actions.columns = ['user_id', 'count']
        re_count = grouped.sum()[['user_id', 'count1']]
        re_count.columns = ['user_id', 're_count']
        actions = pd.merge(actions, re_count, on='user_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['user_id'], re_buy_rate], axis=1)
        actions.columns = ['user_id', 're_buy_rate']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat8_2_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取最近一次行为的时间距离当前时间的差距
def get_action_user_feat9(start_date, end_date):
    dump_path = './cache/user_feat9_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['user_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['user_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat9_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取最后一次行为的次数并且进行归一化
def get_action_user_feat10(start_date, end_date):
    dump_path = './cache/user_feat10_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['user_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["user_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat10_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 获取人物该层级最后一层的各种行为的统计数量
def get_action_user_feat11(start_date, end_date, n):
    dump_path = './cache/user_feat11_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat11_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def get_action_user_feat12(start_date, end_date):
    dump_path = './cache/user_feat12_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['user_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        columns = []
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['user_id', 'level%s' % i, 'time']]
                df = df.groupby(['user_id', 'level%s' % i]).count()
                df = df.unstack()
                df = df.reset_index()
                df.columns = ['user_id'] + list(range(df.shape[1] - 1))
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on='user_id', how='left')
        user_id = result['user_id']
        del result['user_id']
        actions = result.fillna(0)
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions.columns = range(actions.shape[1])
        actions = pd.concat([user_id, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat11_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 层级的天数
def get_action_user_feat13(start_date, end_date, n):
    dump_path = './cache/user_feat13_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['user_id', 'type', 'time'], keep='first')
        actions = df.groupby(['user_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat13_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def get_action_user_feat14(start_date, end_date):
    dump_path = './cache/user_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['user_id', 'time', 'type']]
        df = df[df['type'] == 4][['user_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['user_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['user_id']]
        del actions['user_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat14_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户购买/加入购物车/关注前访问次数
def get_action_user_feat15(start_date, end_date):
    dump_path = './cache/user_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 用户购买前访问次数
        def user_feat_15_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(visit, buy, on='user_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 用户加入购物车前访问次数
        def user_feat_15_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='user_id', how='left')
            actions['visit_num_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 用户关注前访问次数
        def user_feat_15_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('user_id', as_index=False).count()
            visit.columns = ['user_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='user_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车次数
        def user_feat_15_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('user_id', as_index=False).count()
            addtoshopping.columns = ['user_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='user_id', how='left')
            actions['addtoshopping_num_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注次数
        def user_feat_15_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['user_id', 'type']]
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('user_id', as_index=False).count()
            guanzhu.columns = ['user_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('user_id', as_index=False).count()
            buy.columns = ['user_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='user_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(user_feat_15_1(start_date, end_date), user_feat_15_2(start_date, end_date), on='user_id',
                           how='outer')
        actions = pd.merge(actions, user_feat_15_3(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_15_4(start_date, end_date), on='user_id', how='outer')
        actions = pd.merge(actions, user_feat_15_5(start_date, end_date), on='user_id', how='outer')
        user_id = actions['user_id']
        del actions['user_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([user_id, pd.DataFrame(actions)], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat15_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 用户行为的交叉
def get_action_user_feat16(start_date, end_date):
    dump_path = './cache/user_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['user_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('user_id', as_index=False).count()
        del action2['type']
        action2.columns = ['user_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='user_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat16_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 最近k天用户访问P集合的商品数/用户访问总体的商品数（k小于7天，不除总体的商品数，反之，除）
def get_action_user_feat0509_1_30(start_date, end_date, n):
    dump_path = './cache/user_feat0509_1_30_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['user_id', 'sku_id', 'type']]
        actions_dummy = pd.get_dummies(actions['type'], prefix='actions')
        actions = pd.concat([actions, actions_dummy], axis=1)
        del actions['type']

        P = get_basic_product_feat()[['sku_id']]
        P['label'] = 1
        actions_sub = pd.merge(actions, P, on='sku_id', how='left')
        actions_sub = actions_sub[actions_sub['label'] == 1]
        del actions_sub['label']

        actions_sub = actions_sub.groupby(['user_id'], as_index=False).sum()
        del actions_sub['sku_id']
        actions_all = actions.groupby(['user_id'], as_index=False).sum()
        del actions_all['sku_id']

        if n > 7:
            actions = pd.merge(actions_all, actions_sub, on=['user_id'], how='left')
            # print actions.head()
            for i in range(1, 7):
                actions['actions_%s' % i] = actions['actions_%s_y' % i] / actions['actions_%s_x' % i]
                # actions=actions[['user_id','actions_1','actions_2','actions_3','actions_4','actions_5','actions_6']]

        else:
            actions = pd.merge(actions_all, actions_sub, on=['user_id'], how='left')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat30_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 点击模块
def get_action_user_feat0509_1_31(start_date, end_date):
    dump_path = './cache/user_feat0509_1_31_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'model_id']]
        actions['u_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions['u_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions['u_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions['u_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions['u_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby('user_id').sum().reset_index().drop('model_id', axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['user_id'] + ['u_feat31_' + str(i) for i in range(1, actions.shape[1])]
    return actions


#u模型cate=8的购买者和不是cate=8的购买者
def get_action_u0513_feat16(start_date,end_date):
    dump_path = './cache/u0513_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'type', 'cate']]
        df = df[df['type'] == 4]
        df = df.groupby(['user_id', 'cate']).count()
        df = df.unstack().reset_index()
        df.columns = ['user_id'] + ['cate' + str(i) for i in range(4, 12)]
        df = df.fillna(0)
        sum1 = df.drop(['user_id', 'cate8'], axis=1).apply(sum, axis=1)
        sum2 = df.drop(['user_id'], axis=1).apply(sum, axis=1)
        actions = pd.concat([df[['user_id', 'cate8']], sum1, sum2], axis=1)
        actions.columns = ['user_id', 'cate8', 'sum_other_cate', 'sum']
        actions['cate8_rate'] = actions['cate8'] / actions['sum']
        actions['sum_other_cate_rate'] = actions['sum_other_cate'] / actions['sum']
        del actions['sum']
        actions.to_csv(dump_path,index=False)
    return actions

get_action_u0513_feat16('2016-02-01','2016-04-16')
# 用户层级特征
def get_action_user_feat_six_xingwei(start_date, end_date, n):
    dump_path = './cache/user_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("user_zlzl" + str(n))
        return actions
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        num_day = np.max(actions['time'])
        df = None
        print(num_day)
        for i in range(min(num_day + 1, 6)):
            in_temp = pd.get_dummies(actions['type'], prefix="user_action_time_" + str(i))
            temp = actions[actions['time'] == i]
            temp = pd.concat([temp['user_id'], in_temp], axis=1)

            feature = ['user_id']
            for j in range(1, 7, 1):
                feature.append('user_action_time_' + str(i) + '_' + str(j))

            temp = temp.groupby(['user_id'], as_index=False).sum()
            temp.columns = feature
            if df is None:
                df = temp
            else:
                df = pd.merge(df, temp, how='outer', on='user_id')
        df.to_csv(dump_path, index=False)
        return df

def deal_user_six_deal(start_date, end_date, n):
    dump_path = './cache/deal_user_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        actions.columns = ['user_id'] + ['u_featsix_deal_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        return actions
    else:
        temp = get_action_user_feat_six_xingwei(start_date, end_date, n)  # 修改
        time1 = datetime.now()
        columns = ["user_id"]
        all_col = temp.shape[1] - 1
        temp.columns = columns + list(range(all_col))
        temp = temp.fillna(0)
        columns = ['user_id']
        for j in range(0, 6, 1):
            temp["zl_" + str(j)] = 0
            columns.append("zl_" + str(j))
            for k in range(j, all_col, 6):
                temp["zl_" + str(j)] = temp["zl_" + str(j)] + temp[k].map(lambda x: x * ((k // 6 + 1) ** (-0.67)))
            temp["zl_" + str(j)] = temp["zl_" + str(j)].map(lambda x: (x - np.min(temp["zl_" + str(j)])) / (
                np.max(temp["zl_" + str(j)]) - np.min(temp["zl_" + str(j)])))
        temp = temp[columns]
        temp.to_csv(dump_path, index=False)
        return temp

# # get  user sku
# def get_user(start_date, end_date):
#     dump_path = './cache/user_sku_%s_%s.csv' % (start_date, end_date)
#     if os.path.exists(dump_path):
#         actions = pd.read_csv(dump_path)
#     else:
#         actions = get_actions(start_date, end_date)
#         actions = actions[(actions['type'] == 2) | (actions['type'] == 5) | (actions['type'] == 4)]
#         actions=actions[actions['cate']==8]
#         actions = actions[['user_id']]
#         actions = actions.drop_duplicates(['user_id'], keep='first')
#         actions.to_csv(dump_path, index=False)
#     return actions


def get_user_labels(test_start_date,test_end_date):
    dump_path = './cache/user_labels_%s_%s.csv' % (test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(test_start_date, test_end_date)
        actions = actions[actions['cate']==8]
        actions = actions[actions['type'] == 4].drop_duplicates(['user_id'])[['user_id']]
        actions['label'] = 1

    return actions




# 测试集
def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/u_test_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=10)).split(' ')[0]
        actions = get_actions(start_days, train_end_date)
        actions=actions[actions['cate']==8][['user_id']].drop_duplicates(['user_id'])
        print (actions.shape)
        start_days = "2016-02-01"
        actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
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
        actions = pd.merge(actions, get_action_u0513_feat16(start_days, train_end_date), how='left', on='user_id')
        print (actions.shape)
        for i in (1, 2, 3, 7, 14, 28):
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
        print(actions.shape)

        actions = actions.fillna(0)


    return actions


# 训练集
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date):
    dump_path = './cache/u_train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        start_days=str(pd.to_datetime(train_end_date)-timedelta(days=10)).split(' ')[0]
        actions = get_actions(start_days, train_end_date)
        actions=actions[actions['cate']==8][['user_id']].drop_duplicates(['user_id'])
        print (actions.shape)
        start_days = "2016-02-01"
        actions = pd.merge(actions,get_basic_user_feat() , how='left', on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_action_user_feat1(start_days, train_end_date), how='left', on='user_id')
        print(actions.shape)
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
        actions=pd.merge(actions,get_action_u0513_feat16(start_days,train_end_date),how='left',on='user_id')
        print (actions.shape)
        for i in (1, 2, 3, 7, 14, 28):
            actions = pd.merge(actions, deal_user_six_deal(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat11(train_start_date, train_end_date, i), how='left',on='user_id')
            actions = pd.merge(actions, get_action_user_feat13(train_start_date, train_end_date, i), how='left',on='user_id')
        print(actions.shape)
        actions = pd.merge(actions, get_user_labels(test_start_date, test_end_date), how='left', on='user_id')
        actions = actions.fillna(0)
        print(actions.shape)
    return  actions


