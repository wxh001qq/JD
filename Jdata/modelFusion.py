# -*- coding: utf-8 -*-

"""
求daoliker和ottsion的预测结果的交集
"""
import pandas as pd


#　answer_my = pd.read_csv('HS_0.10392+0.11089_0.14718_0.08669.csv', header=0)
# answer_my = pd.read_csv('rizi0506001.csv', header=0)
# answer_my = pd.read_csv('rizi0508001.csv', header=0)
# answer_HS = pd.read_csv('HS_503F001.csv', header=0)
# answer_qingchun = pd.read_csv('qingchun0508002.csv', header=0)
# 交换一下my和qingchun
answer_qingchun = pd.read_csv('rizi0508001.csv', header=0)
answer_HS = pd.read_csv('HS_503F001.csv', header=0)
answer_my = pd.read_csv('qingchun0508002.csv', header=0)


# merge后去重(每个user_id只能买一个sku_id)
# answer = answer_my.append(answer_HS)
# answer = answer.reset_index(drop=True)
# answer = answer.drop_duplicates()
# answer.to_csv('answer_HS_zlp.csv', index=False, encoding='gbk')

answer_my_userid = list(answer_my.iloc[:, 0])
answer_HS_userid = list(answer_HS.iloc[:, 0])
interSec = list(set(answer_my_userid).intersection(set(answer_HS_userid)))
Union = list(set(answer_my_userid).union(set(answer_HS_userid)))

# mask_1 = answer_my['user_id'].isin(interSec)
# answer_my_interSec = answer_my[mask_1]
# answer_my_interSec.to_csv('answer_my_interSec.csv', index=False, encoding='gbk')
#
# mask_2 = answer_HS['user_id'].isin(interSec)
# # answer_HS['user_id'] = [int(i) for i in answer_HS['user_id']]
# answer_HS_interSec = answer_HS[mask_2]
# answer_HS_interSec.to_csv('answer_HS_interSec.csv',  index=False, encoding='gbk')

# # 将两者的user_id交集中sku_id不同的去掉
# list = list(answer_my['user_id'])
# for i in answer_HS['user_id']:    # 297250
#     if i not in list:
#         if answer_HS[answer_HS['user_id'] == i]['sku_id'] != answer_my[answer_my['user_id'] == i]['sku_id']
#         answer_my = answer_my.append(answer_HS[answer_HS['user_id'] == i])
#         answer_my = answer_my.reset_index(drop=True)
#     else:
#         answer_my = answer_my.drop(answer_HS[answer_HS['user_id'] == i].index)
#
# answer_my.to_csv('answer_HS_zlp_noIntersec.csv', index=False, encoding='gbk')

# 并集, HS和zlp
list_1 = list(answer_my['user_id'])
for i in answer_HS['user_id']:    # 297250
    if i not in list_1:
        answer_my = answer_my.append(answer_HS[answer_HS['user_id'] == i])
        answer_my = answer_my.reset_index(drop=True)

answer_my.to_csv('answer_HS_zlp.csv', index=False, encoding='gbk')

# 并集，HS、zlp和qingchun
answer_HS_zlp = pd.read_csv('answer_HS_zlp.csv', header=0)
list_2 = list(answer_HS_zlp['user_id'])
for i in answer_qingchun['user_id']:    # 297250
    if i not in list_2:
        answer_HS_zlp = answer_HS_zlp.append(answer_qingchun[answer_qingchun['user_id'] == i])
        answer_HS_zlp = answer_HS_zlp.reset_index(drop=True)

answer_HS_zlp.to_csv('answer_HS_zlp_qingchun.csv', index=False, encoding='gbk')

# # 两者都有的，用HS的sku_id
# list = list(answer_HS['user_id'])
# for i in answer_my['user_id']:    # 297250
#     if i not in list:
#         answer_HS = answer_HS.append(answer_my[answer_my['user_id'] == i])
#         answer_HS = answer_HS.reset_index(drop=True)
#
# answer_HS.to_csv('answer_HS_zlp_HSsku_id.csv', index=False, encoding='gbk')

# list = list(answer_HS['user_id'])
# for i in answer_my['user_id']:    # 297250
#     if i in list:
#         answer_HS = answer_HS.drop(answer_HS[answer_HS['user_id'] == i].index)
#         answer_HS = answer_HS.append(answer_my[answer_my['user_id'] == i])
#         answer_HS = answer_HS.reset_index(drop=True)
#
# answer_HS.to_csv('answer_HS_zlp_interSec_replace.csv', index=False, encoding='gbk')

# mask_2 = answer_HS['user_id'].isin(interSec)
# # answer_HS['user_id'] = [int(i) for i in answer_HS['user_id']]
# answer_HS_interSec = answer_HS[mask_2]
# answer_HS_interSec.to_csv('answer_HS_interSec.csv',  index=False, names=['user_id', 'sku_id'], encoding='gbk')

# mask_1 = answer_my['user_id'].isin(interSec)
# answer_rf_interSec = answer_rf[mask_1]
# answer_rf_interSec.to_csv('answer_rf_interSec.csv', index=False, names=['user_id', 'sku_id'], encoding='gbk')
#
# mask_2 = answer_gbdt['user_id'].isin(interSec)
# answer_gbdt['user_id'] = [int(i) for i in answer_gbdt['user_id']]
# answer_gbdt_interSec = answer_gbdt[mask_2]
# answer_gbdt_interSec.to_csv('answer_gbdt_interSec.csv',  index=False, names=['user_id', 'sku_id'], encoding='gbk')

# if __name__ == __main__:
