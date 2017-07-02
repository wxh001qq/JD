#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#from basic_feat0518 import *


# top k 天的行为次数总和(滑窗处理)
def get_action_p0509_feat(start_date, end_date, k):
    dump_path = './cache/p_action_%s_%s_%s.csv' % (start_date, end_date, k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_actions(start_days, end_date)
        actions = actions[['sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='type')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby('sku_id', as_index=False).sum()
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(actions.drop(['sku_id', 'type'], axis=1).values)
        df = pd.DataFrame(df)

        actions = pd.concat([actions[['sku_id']], df], axis=1)
        actions.columns=['sku_id']+['p0509_' + str(k) + '_' + str(i) for i in range(1, actions.shape[1])]
        actions.to_csv(dump_path, index=False)
    return actions

#print get_action_p0509_feat('2016-02-01','2016-04-11',1)

# 商品的行为转化率
def get_action_product_feat_1(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = './cache/product_feat_accumulate_xiugai_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        #         actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_3_ratio'] = actions['action_3'] / actions['action_2']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat1_' + str(i) for i in range(1, actions.shape[1])]
    return actions

# 商品购买/加入购物车/关注前访问天数
def get_action_p0509_feat_2(start_date, end_date):
    dump_path = './cache/product_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 商品购买前访问天数
        def product_feat_2_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(visit, buy, on='sku_id', how='left')
            actions['visit_day_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品加入购物车前访问天数
        def product_feat_2_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='sku_id', how='left')
            actions['visit_day_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 商品关注前访问天数
        def product_feat_2_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            visit = actions[actions['type'] == 1]
            visit = visit.drop_duplicates(['sku_id', 'time'], keep='first')
            del visit['time']
            del actions['time']
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='sku_id', how='left')
            actions['visit_day_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车天数
        def product_feat_2_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            # actions=actions.drop_duplicates(['user_id','time'],keep='first')
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.drop_duplicates(['sku_id', 'time'], keep='first')
            del addtoshopping['time']
            del actions['time']
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='sku_id', how='left')
            actions['addtoshopping_day_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注天数
        def product_feat_2_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
            actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.drop_duplicates(['sku_id', 'time'], keep='first')
            del guanzhu['time']
            del actions['time']
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='sku_id', how='left')
            actions['guanzhu_day_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(product_feat_2_1(start_date, end_date), product_feat_2_2(start_date, end_date),
                           on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_3(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_4(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, product_feat_2_5(start_date, end_date), on='sku_id', how='outer')
        sku_id = actions['sku_id']
        del actions['sku_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat2_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 商品平均访问间隔
def get_action_p0509_feat_6(start_date, end_date):
    dump_path = './cache/product_feat7_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: x.split(' ')[0])
        df = df.drop_duplicates(['sku_id', 'time'], keep='first')
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        actions = df.groupby('sku_id', as_index=False).agg(lambda x: x['time'].diff().mean())
        actions['avg_visit'] = actions['time'].dt.days
        del actions['time']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat6_' + str(i) for i in range(1, actions.shape[1])]
    return actions

# 商品六种行为平均访问间隔
def get_action_p0509_feat_6_six(start_date, end_date):
    dump_path = './cache/product_feat7_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: (-1) * get_day_chaju(x, start_date))
        df = df.drop_duplicates(['sku_id', 'time', 'type'], keep='first')
        actions = df.groupby(['sku_id', 'type']).agg(lambda x: np.diff(x).mean())
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat6_six_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 最近K天
def product_top_k_0_1(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    actions = actions.groupby('sku_id', as_index=False).sum()
    del actions['type']
    del actions['user_id']
    sku_id = actions['sku_id']
    del actions['sku_id']
    actions = actions.applymap(lambda x: 1 if x > 0 else 0)
    actions = pd.concat([sku_id, actions], axis=1)
    return actions


# 最近K天行为0/1提取
def get_action_p0509_feat_8(start_date, end_date):
    dump_path = './cache/product_feat9_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 6, 7, 15, 30):
            print(i)
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = product_top_k_0_1(start_days, end_date)
            else:
                actions = pd.merge(actions, product_top_k_0_1(start_days, end_date), how='outer', on='sku_id')
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat8_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 商品的重复购买率
def get_action_p0509_feat_8_2(start_date, end_date):
    dump_path = './cache/product_feat8_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        df = df[df['type'] == 4]  # 购买的行为
        df = df.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'count1']
        df['count1'] = df['count1'].map(lambda x: 1 if x > 1 else 0)
        grouped = df.groupby(['sku_id'], as_index=False)
        actions = grouped.count()[['sku_id', 'count1']]
        actions.columns = ['sku_id', 'count']
        re_count = grouped.sum()[['sku_id', 'count1']]
        re_count.columns = ['sku_id', 're_count']
        actions = pd.merge(actions, re_count, on='sku_id', how='left')
        re_buy_rate = actions['re_count'] / actions['count']
        actions = pd.concat([actions['sku_id'], re_buy_rate], axis=1)
        actions.columns = ['sku_id', 're_buy_rate']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat8_2_' + str(i) for i in range(1, actions.shape[1])]
    return actions




# 获取货物最近一次行为的时间距离当前时间的差距
def get_action_p0509_feat_9(start_date, end_date):
    dump_path = './cache/product_feat9_2_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        # df['time'] = df['time'].map(lambda x: (-1)*get_day_chaju(x,start_date))
        df = df.drop_duplicates(['sku_id', 'type'], keep='last')
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)
        actions = df.groupby(['sku_id', 'type']).sum()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(30)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat9_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取货品最后一次行为的次数并且进行归一化
def get_action_product_feat_10(start_date, end_date):
    dump_path = './cache/product_feat10_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:

        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) + 1)

        idx = df.groupby(['sku_id', 'type'])['time'].transform(min)
        idx1 = idx == df['time']
        actions = df[idx1].groupby(["sku_id", "type"]).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()

        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat10_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取物品该层级最后一层的各种行为的统计数量
def get_action_product_feat_11(start_date, end_date, n):
    dump_path = './cache/product_feat11_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df[df['time'] == 0]
        del df['time']
        temp = pd.get_dummies(df['type'], prefix='type')
        del df['type']
        actions = pd.concat([df, temp], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat11_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions



def get_action_product_feat_12(start_date, end_date):
    dump_path = './cache/p0509_feat12_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        actions['time'] = actions['time'].map(lambda x: x.split(' ')[0])
        actions = actions.drop_duplicates(['sku_id', 'time', 'type'], keep='first')
        actions['day'] = actions['time'].map(
            lambda x: (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)
        result = None
        for i in (2, 3, 7, 14, 28):  # 层级个数
            print ('i%s' % i)
            actions['level%s' % i] = actions['day'].map(lambda x: x // i)
            a = set(actions['level%s' % i].tolist())
            for j in (1, 2, 3, 4, 5, 6):  # type
                print ('j%s' % j)
                df = actions[actions['type'] == j][['sku_id', 'level%s' % i, 'time']]
                df = df.groupby(['sku_id', 'level%s' % i]).count()
                df = df.unstack()
                b = df.columns.levels[1].tolist()
                df.columns = ['p_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k) for k in
                              df.columns.levels[1].tolist()]
                if len(list(a - set(b))) != 0:
                    c = list(a - set(b))
                    for k in c:
                        df['p_feat12_' + str('level%s_' % i) + str(j) + '_' + str(k)] = 0
                columns = df.columns
                dict = {}
                for column in columns:
                    k = int(column.split('_')[-1])
                    dict[column] = k
                columns = sorted(dict.items(), key=lambda x: x[1])
                columns = [(columns[t])[0] for t in range(len(columns))]
                df = df[columns]
                df = df.reset_index()
                if result is None:
                    result = df
                else:
                    result = pd.merge(result, df, on='sku_id', how='left')
        columns = result.columns
        sku_id = result['sku_id']
        del result['sku_id']
        actions = result.fillna(0)

        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([sku_id, actions], axis=1)
        actions.columns = columns
        actions.to_csv(dump_path, index=False)
    return actions





# 层级天数
def get_action_product_feat_13(start_date, end_date, n):
    dump_path = './cache/sku_feat13_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        df = df.drop_duplicates(['sku_id', 'type', 'time'], keep='first')
        actions = df.groupby(['sku_id', 'type']).count()
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.fillna(0)
        actions = actions.reset_index()
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat13_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions


def get_action_product_feat_14(start_date, end_date):
    dump_path = './cache/sku_feat14_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        n = 5
        df = get_actions(start_date, end_date)[['sku_id', 'time', 'type']]
        df = df[df['type'] == 4][['sku_id', 'time']]
        df['time'] = df['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        days = np.max(df['time'])

        df['cnt'] = 0
        actions = df.groupby(['sku_id', 'time']).count()

        actions = actions.unstack()

        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()

        actions = actions.fillna(0)
        user_sku = actions[['sku_id']]
        del actions['sku_id']
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([user_sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat14_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 商品购买/加入购物车/关注前访问次数
def get_action_p0509_feat_15(start_date, end_date):
    dump_path = './cache/p0509_feat15_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        # 商品购买前访问次数
        def p0509_feat_15_1(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(visit, buy, on='sku_id', how='left')
            actions['visit_num_before_buy'] = actions['visit'] / actions['buy']
            del actions['buy']
            del actions['visit']
            return actions

        # 商品加入购物车前访问次数
        def p0509_feat_15_2(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            actions = pd.merge(visit, addtoshopping, on='sku_id', how='left')
            actions['visit_num_before_addtoshopping'] = actions['visit'] / actions['addtoshopping']
            del actions['addtoshopping']
            del actions['visit']
            return actions

        # 商品关注前访问次数
        def p0509_feat_15_3(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            visit = actions[actions['type'] == 1]
            visit = visit.groupby('sku_id', as_index=False).count()
            visit.columns = ['sku_id', 'visit']
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            actions = pd.merge(visit, guanzhu, on='sku_id', how='left')
            actions['visit_num_before_guanzhu'] = actions['visit'] / actions['guanzhu']
            del actions['guanzhu']
            del actions['visit']
            return actions

        # 用户购买前加入购物车次数
        def p0509_feat_15_4(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            addtoshopping = actions[actions['type'] == 2]
            addtoshopping = addtoshopping.groupby('sku_id', as_index=False).count()
            addtoshopping.columns = ['sku_id', 'addtoshopping']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(addtoshopping, buy, on='sku_id', how='left')
            actions['addtoshopping_num_before_buy'] = actions['addtoshopping'] / actions['buy']
            del actions['buy']
            del actions['addtoshopping']
            return actions

        # 用户购买前关注次数
        def p0509_feat_15_5(start_date, end_date):
            actions = get_actions(start_date, end_date)[['sku_id', 'type']]
            guanzhu = actions[actions['type'] == 5]
            guanzhu = guanzhu.groupby('sku_id', as_index=False).count()
            guanzhu.columns = ['sku_id', 'guanzhu']
            buy = actions[actions['type'] == 4]
            buy = buy.groupby('sku_id', as_index=False).count()
            buy.columns = ['sku_id', 'buy']
            actions = pd.merge(guanzhu, buy, on='sku_id', how='left')
            actions['guanzhu_num_before_buy'] = actions['guanzhu'] / actions['buy']
            del actions['buy']
            del actions['guanzhu']
            return actions

        actions = pd.merge(p0509_feat_15_1(start_date, end_date), p0509_feat_15_2(start_date, end_date), on='sku_id',
                           how='outer')
        actions = pd.merge(actions, p0509_feat_15_3(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, p0509_feat_15_4(start_date, end_date), on='sku_id', how='outer')
        actions = pd.merge(actions, p0509_feat_15_5(start_date, end_date), on='sku_id', how='outer')
        sku_id = actions['sku_id']
        del actions['sku_id']
        actions = actions.fillna(0)
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions)], axis=1)

        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat15_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 商品行为的交叉
def get_action_product_feat_16(start_date, end_date):
    dump_path = './cache/product_feat16_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['sku_id', 'type']]
        actions['cnt'] = 0
        action1 = actions.groupby(['sku_id', 'type']).count()
        action1 = action1.unstack()
        index_col = list(range(action1.shape[1]))
        action1.columns = index_col
        action1 = action1.reset_index()
        action2 = actions.groupby('sku_id', as_index=False).count()
        del action2['type']
        action2.columns = ['sku_id', 'cnt']
        actions = pd.merge(action1, action2, how='left', on='sku_id')
        for i in index_col:
            actions[i] = actions[i] / actions['cnt']
        del actions['cnt']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat16_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# 老顾客比率
def get_action_p0509_feat_17(start_date, end_date):
    dump_path = './cache/product_feat4_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)[['user_id', 'sku_id', 'type']]
        actions = actions[actions['type'] == 4]
        df = actions.groupby(['user_id', 'sku_id'], as_index=False).count()
        df.columns = ['user_id', 'sku_id', 'number']
        df2 = df[df['number'] > 1]
        del df['number']
        del df2['number']
        df1 = df.groupby('sku_id', as_index=False).count()
        df1.columns = ['sku_id', 'all_number']
        df2 = df2.groupby('sku_id', as_index=False).count()
        df2.columns = ['sku_id', 'number']
        actions = pd.merge(df1, df2, on='sku_id', how='left')
        actions['rebuy_rate'] = actions['number'] / actions['all_number']
        del actions['number']
        del actions['all_number']
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat17_' + str(i) for i in range(1, actions.shape[1])]
    return actions

#get_action_p0509_feat_18

# 商品点击到购买的时间间隔
def get_action_p0509_feat_19(start_date, end_date):
    dump_path = './cache/p0509_feat_19_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        actions_dianji = actions[actions['type'] == 6][['user_id', 'sku_id', 'time']]
        actions_dianji['time_dianji'] = actions_dianji['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_dianji = actions_dianji[['user_id', 'sku_id', 'time_dianji']]
        actions_dianji = actions_dianji.drop_duplicates(['user_id', 'sku_id'], keep='first')

        actions_goumai = actions[actions['type'] == 4][['user_id', 'sku_id', 'time']]
        actions_goumai['time_goumai'] = actions_goumai['time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        actions_goumai = actions_goumai[['user_id', 'sku_id', 'time_goumai']]
        actions_goumai = actions_goumai.drop_duplicates(['user_id', 'sku_id'], keep='last')

        actions = pd.merge(actions_dianji, actions_goumai, on=['user_id', 'sku_id'], how='inner')
        actions['time_jiange'] = actions['time_goumai'] - actions['time_dianji']
        actions = actions.drop(['user_id', 'time_goumai', 'time_dianji'], axis=1)
        actions['time_jiange'] = actions['time_jiange'].map(lambda x: x.days * 24 + x.seconds // 3600 + 1)

        actions_min = actions.groupby('sku_id').min().reset_index()
        actions_min.columns = ['sku_id', 'time_min']
        # actions_mean = actions.groupby('user_id').mean().reset_index()
        # actions_mean.columns = ['user_id','time_mean']
        actions_max = actions.groupby('sku_id').max().reset_index()
        actions_max.columns = ['sku_id', 'time_max']
        actions = pd.merge(actions_min, actions_max, on='sku_id', how='left')

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        actions=actions.astype('float')
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat19_' + str(i) for i in range(1, actions.shape[1])]
    return actions




# 获取某商品某段时间内加入购物车的数量以及关注的数量
def get_action_p0509_feat_21(start_date, end_date, n):
    dump_path = './cache/p0509_feat_21_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')

        actions = get_actions(start_days, end_date)[['sku_id', 'type', 'cate']]
        actions_gouwuche = actions[actions['type'] == 2]
        actions_gouwuche_1 = actions_gouwuche[['sku_id', 'type']]
        actions_gouwuche_1 = actions_gouwuche_1.groupby('sku_id').count().reset_index()
        actions_gouwuche_1.columns = ['sku_id', str(n) + 'gouwuche_add']

        actions_gouwuche = actions_gouwuche[actions_gouwuche['cate'] == 8]
        actions_gouwuche_2=actions_gouwuche[['sku_id', 'type']]

        actions_gouwuche_2 = actions_gouwuche_2.groupby('sku_id').count().reset_index()
        actions_gouwuche_2.columns = ['sku_id', str(n) + 'gouwuche_add_cate_8']

        actions_guanzhu = actions[actions['type'] == 5]
        actions_guanzhu_1 = actions_guanzhu[['sku_id', 'type']]
        actions_guanzhu_1 = actions_guanzhu_1.groupby('sku_id').count().reset_index()
        actions_guanzhu_1.columns = ['sku_id', str(n) + 'guanzhu_add']

        actions_guanzhu = actions_guanzhu[actions_guanzhu['cate'] == 8]
        actions_guanzhu_2=actions_guanzhu[['sku_id', 'type']]
        actions_guanzhu_2 = actions_guanzhu_2.groupby('sku_id').count().reset_index()
        actions_guanzhu_2.columns = ['sku_id', str(n) + 'guanzhu_add_cate_8']

        actions = pd.merge(actions_gouwuche_1, actions_gouwuche_2, on='sku_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_1, on='sku_id', how='outer')
        actions = pd.merge(actions, actions_guanzhu_2, on='sku_id', how='outer')
        actions = actions.fillna(0)

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        actions=actions.astype('float')
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    return actions





# 商品总购买/加购/关注/点击/浏览品牌数
def get_action_p0509_feat5(start_date, end_date):
    dump_path = './cache/p0509_feat5_a_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(start_date, end_date)
        action = None
        for i in (1, 2, 4, 5, 6):
            df = actions[actions['type'] == i][['user_id', 'sku_id']]
            df = df.drop_duplicates(['user_id', 'sku_id'], keep='first')
            df = df.groupby('sku_id', as_index=False).count()
            df.columns = ['sku_id', 'num_%s' % i]
            if i == 1:
                action = df
            else:
                action = pd.merge(action, df, on='sku_id', how='outer')
        actions = action.fillna(0)
        sku = actions[['sku_id']]
        min_max_scaler = preprocessing.MinMaxScaler()
        actions = min_max_scaler.fit_transform(actions.drop(['sku_id'], axis=1).values)
        actions = pd.DataFrame(actions)
        actions = pd.concat([sku, actions], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat5_' + str(i) for i in range(1, actions.shape[1])]
    return actions


# top  k 商品总购买/加购/关注/点击/浏览品牌数
def get_action_p0509_feat_23(start_date, end_date, k):
    dump_path = './cache/p0509_feat23_%s_%s_%s.csv' % (start_date, end_date, k)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = pd.to_datetime(end_date) - timedelta(days=k)
        start_days = str(start_days).split(' ')[0]
        actions = get_action_p0509_feat5(start_days, end_date)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p0509_feat_23_' + str(k) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 点击模块
def get_action_p0509_feat_24(start_date, end_date, n):
    dump_path = './cache/p0509_feat_24_%s_%s_%s.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=n)
        start_days = datetime.strftime(start_days, '%Y-%m-%d')
        actions = get_actions(start_days, end_date)
        actions = actions[actions['type'] == 6][['sku_id', 'model_id']]

        actions_click_sum = actions[['sku_id', 'model_id']].groupby('sku_id').count().reset_index()
        actions_click_sum.columns = ['sku_id', str(n) + 'click_sum_all']
        actions[str(n) + 'p_click14_history'] = actions['model_id'].map(lambda x: int(x == 14))
        actions[str(n) + 'p_click21_history'] = actions['model_id'].map(lambda x: int(x == 21))
        actions[str(n) + 'p_click28_history'] = actions['model_id'].map(lambda x: int(x == 28))
        actions[str(n) + 'p_click110_history'] = actions['model_id'].map(lambda x: int(x == 110))
        actions[str(n) + 'p_click210_history'] = actions['model_id'].map(lambda x: int(x == 210))
        actions = actions.groupby('sku_id').sum().reset_index().drop('model_id', axis=1)

        actions = pd.merge(actions, actions_click_sum, how='left', on='sku_id')

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

        sku_id = actions[['sku_id']]
        del actions['sku_id']
        actions = actions.fillna(0)
        columns = actions.columns
        min_max_scale = preprocessing.MinMaxScaler()
        actions = min_max_scale.fit_transform(actions.values)
        actions = pd.concat([sku_id, pd.DataFrame(actions, columns=columns)], axis=1)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p0509_feat_24_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
    return actions



# 获取每个商品的被购买的六种行为get_action_sku_feat_six_xingwei
def get_action_p0509_feat_26(start_date, end_date, n):
    dump_path = './cache/sku_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("sku_zlzl_" + str(n))
        return actions
    else:
        actions = get_actions(start_date, end_date)

        actions['time'] = actions['time'].map(lambda x: get_day_chaju(x, end_date) // n)
        num_day = np.max(actions['time'])
        df = None
        #         print(num_day)
        for i in range(min(num_day + 1, 6)):

            in_temp = pd.get_dummies(actions['type'], prefix="sku_action_time_" + str(i))
            temp = actions[actions['time'] == i]
            temp = pd.concat([temp['sku_id'], in_temp], axis=1)

            feature = ['sku_id']
            for j in range(1, 7, 1):
                feature.append('sku_action_time_' + str(i) + '_' + str(j))

            temp = temp.groupby(['sku_id'], as_index=False).sum()
            #             temp['user_id']=temp['user_id'].astype('int')
            temp.columns = feature
            #             print(temp)
            #           用于归一化
            #             for j in range(1,7,1):
            # #                 min_x=np.min(temp['user_action_time_'+str(j)+'_'+str(i)])
            # #                 max_x=np.max(temp['user_action_time_'+str(j)+'_'+str(i)])
            #                 temp['sku_action_time_'+str(i)+'_'+str(j)]=temp['sku_action_time_'+str(i)+'_'+str(j)].map(lambda x: (x - np.min(temp['sku_action_time_'+str(i)+'_'+str(j)])) / (np.max(temp['sku_action_time_'+str(i)+'_'+str(j)])-np.min(temp['sku_action_time_'+str(i)+'_'+str(j)])))
            if df is None:
                df = temp
            else:
                df = pd.merge(df, temp, how='outer', on='sku_id')
        df.to_csv(dump_path, index=False)
        return df

#deal_sku_six_deal
def get_action_p0509_feat_27(start_date, end_date, n):
    dump_path = './cache/deal_sku_six_action_%s_%s_%s_int.csv' % (start_date, end_date, n)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
        print("wxl_" + str(n))
        actions.columns = ['sku_id'] + ['p_featsixdeal_' + str(n) + '_' + str(i) for i in range(1, actions.shape[1])]
        return actions
    else:
        sku_temp = get_action_p0509_feat_26(start_date, end_date, n)  # 修改
        columns = ["sku_id"]
        all_col = sku_temp.shape[1] - 1
        #         print(sku_temp.head(10))
        sku_temp.columns = columns + list(range(all_col))
        #         print(sku_temp.head(10))
        sku_temp = sku_temp.fillna(0)
        columns = ['sku_id']
        for j in range(0, 6, 1):
            sku_temp["zl_" + str(j)] = 0
            columns.append("zl_" + str(j))
            for k in range(j, all_col, 6):
                #                 print(sku_temp[k].head(1))
                #                 print(sku_temp["zl_"+str(j)].head(1))
                sku_temp["zl_" + str(j)] = sku_temp["zl_" + str(j)] + sku_temp[k].map(
                    lambda x: x * ((k // 6 + 1) ** (-0.67)))
            # print(sku_temp["zl_"+str(j)].head(1))
            sku_temp["zl_" + str(j)] = sku_temp["zl_" + str(j)].map(lambda x: (x - np.min(sku_temp["zl_" + str(j)])) / (
                np.max(sku_temp["zl_" + str(j)]) - np.min(sku_temp["zl_" + str(j)])))
        sku_temp = sku_temp[columns]
        sku_temp.to_csv(dump_path, index=False)
        return sku_temp



# 商品的六种行为的频率
def get_action_p0509_feat_28(start_date, end_date):
    dump_path = './cache/user_feat7_2_six_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        df = get_actions(start_date, end_date)[['sku_id', 'type', 'time']]
        actions = df.groupby(['sku_id', 'type'], as_index=False).count()
        time_min = df.groupby(['sku_id', 'type'], as_index=False).min()
        time_max = df.groupby(['sku_id', 'type'], as_index=False).max()

        time_cha = pd.merge(time_max, time_min, on=['sku_id', 'type'], how='left')
        time_cha['time_x'] = time_cha['time_x'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        time_cha['time_y'] = time_cha['time_y'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

        time_cha['cha_hour'] = 1 + (time_cha['time_x'] - time_cha['time_y']).dt.days * 24 + (time_cha['time_x'] -
                                                                                             time_cha[
                                                                                                 'time_y']).dt.seconds // 3600
        del time_cha['time_x']
        del time_cha['time_y']
        # time_cha=time_cha.fillna(1)

        actions = pd.merge(time_cha, actions, on=['sku_id', 'type'], how="left")
        actions = actions.groupby(['sku_id', 'type']).sum()
        actions['cnt/time'] = actions['time'] / actions["cha_hour"]
        actions = actions.unstack()
        actions.columns = list(range(actions.shape[1]))
        actions = actions.reset_index()
        actions = actions.fillna(0)
        actions.to_csv(dump_path, index=False)
    actions.columns = ['sku_id'] + ['p_feat_28_' + str(i) for i in range(1, actions.shape[1])]
    return actions
print("finish")
