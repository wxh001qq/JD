#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import math
import numpy as np
from sklearn import preprocessing
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

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
    return actions

# 评论数据
comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]
def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pd.read_csv(dump_path)
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
        # del comments['dt']
        # del comments['comment_num']
        comments = comments[
            ['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3',
             'comment_num_4']]
        comments.to_csv(dump_path, index=False)
    return comments


# 获取两个时间相差几天
def get_day_chaju(x, end_date):
    #     x=x.split(' ')[0]
    x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    return (end_date - x).days
print("finish")