#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:34:39 2019
股票型基金和混合型基金业绩评价，进行基金优选
topsis序逻辑
给张芳的结果-股票和混合在一起统计 ，年化收益率给2倍权重
其他给1倍权重
@author: yeecall
"""

from __future__ import division
import pandas as pd
import numpy as np
# import os
# py_path=r'/Users/yeecall/Documents/mywork/joinquant_data/基金评价'
# os.chdir(py_path)
import matplotlib.pyplot as plt
import math
import datetime
from dateutil.parser import parse

# from statsmodels import regression
# import statsmodels.api as sm
from jqdatasdk import *

auth('18610039264', 'zg19491001')
count_ = get_query_count()
print(count_)


# 提取符合条件的基金名单
def fund_find(start_day, operate_mode, underlying_asset_type):
    q = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.operate_mode_id == operate_mode,
                                             finance.FUND_MAIN_INFO.underlying_asset_type_id == underlying_asset_type,
                                             finance.FUND_MAIN_INFO.start_date < start_day)
    df = finance.run_query(q)
    print('一共' + str(len(df)) + '只基金')
    return (df)


def etf_find(s_date, e_date):
    df = get_all_securities(types=['etf', 'fja', 'fjb'])
    df = df[(df['start_date'] < s_date) & (df['end_date'] > e_date)]
    return df


# 提取基金净值数据
def fund_value(start_day, code):
    q = query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value,
              finance.FUND_NET_VALUE.refactor_net_value).filter(finance.FUND_NET_VALUE.code == code,
                                                                finance.FUND_NET_VALUE.day > start_day)
    df = finance.run_query(q)
    return (df)


# 获取价格
def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=False, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


# 收益类指标
# 年化收益率
def annROR(netlist, days):
    '''
    :param netlist:净值曲线
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 252 / days) - 1


# 超额收益率
def over_annROR(lst1, lst2):
    '''
    :lst1:基准指数净值数据
     lst2:基金净值数据
    :return:超额收益率
    '''
    yoy1 = lst1[-1] / lst1[0]
    yoy2 = lst2[-1] / lst2[0]
    return (math.pow(yoy2 / yoy1, 252 / len(lst1)) - 1)


# 风险类指标
# 最大回撤率
def maxRetrace(list):
    '''
    :param list:netlist
    :return: 最大历史回撤
    '''
    Max = 0.0001
    for i in range(len(list)):
        if 1 - list[i] / max(list[:i + 1]) > Max:
            Max = 1 - list[i] / max(list[:i + 1])

    return Max


# 平均回撤率
def avgRetrace(lst):
    '''
    :param list:netlist
    :return: 平均回撤率
    '''
    everyRetrace = list()
    for i in range(len(lst) - 1):
        t = lst[:(len(lst) - i)]
        t_max = max(t)
        retrace = ((t_max - t[-1]) / t_max) * (0.9 ** (i))
        everyRetrace.append(retrace)
    return (sum(everyRetrace))


# Var 
def Var(lst, a=0.01):
    '''
    :param list:netlist
    :return: 平均回撤率
    '''
    from scipy.stats import norm

    llst = [np.log(i) for i in lst]
    llst_0 = llst[:-1]
    llst_1 = llst[1:]
    lr = list()
    for i in range(len(llst_1)):
        lr.append(llst_1[i] - llst_0[i])
    m = np.mean(lr)
    d = np.std(lr)
    var = norm.ppf(a) * d + m
    return (var)


# 收益风险指标
# 夏普比率
def yearsharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


# 索提诺比率
def SortinoRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    op_row = [i for i in row if i < 0]
    return np.mean(row) / np.std(op_row) * math.pow(252, 0.5)


# 信息比率IR
def IR(netlist, days):
    return (annROR(netlist, days) / maxRetrace(netlist))


def topsis(df, zb_lst, direc, w):
    temp = df.copy()
    zb_list = zb_lst
    tmp = temp[zb_list]
    tmp1 = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    g = list()
    b = list()
    for i in range(len(direc)):
        if direc[i] > 0:
            g.append(tmp1.iloc[:, i].max())
            b.append(tmp1.iloc[:, i].min())
        else:
            g.append(tmp1.iloc[:, i].min())
            b.append(tmp1.iloc[:, i].max())
    g_d = pd.Series(g, index=zb_list)
    b_d = pd.Series(b, index=zb_list)
    G_tmp = tmp1 - g_d
    tmp1['G'] = G_tmp.apply(
        lambda x: w[0] * x['annROR'] ** 2 + w[1] * x['avgRetrace'] ** 2 + w[2] * x['Var'] ** 2 + w[3] * x['IR'] ** 2,
        axis=1)
    B_tmp = tmp1 - b_d
    tmp1['B'] = B_tmp.apply(
        lambda x: w[0] * x['annROR'] ** 2 + w[1] * x['avgRetrace'] ** 2 + w[2] * x['Var'] ** 2 + w[3] * x['IR'] ** 2,
        axis=1)
    temp['topsis'] = tmp1['B'] / (tmp1['G'] + tmp1['B'])
    return temp


if __name__ == '__main__':
    start = '2017-01-01'
    fold = 'e:/fof/'
    today = str(datetime.datetime.today())[:10]
    days = int((parse(today) - parse(start)).days * (252 / 365))
    operate_mode_id = [401001, 401003, 401005, 401006]
    underlying_asset_type_id = [402001, 402003, 402004, 402005]
    # fund_id 为符合条件的基金名单
    num = 0
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            if num == 0:
                fund_id = fund_find(start, i, j)
            else:
                fund_id = pd.concat([fund_id, fund_find(start, i, j)])
            num += 1
    fund_id.to_csv(fold + 'funds_zf.csv', encoding='gbk')
    # fund_id = pd.read_csv(fold + 'funds_zf.csv', encoding='gbk')
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    # fund_id['find']=fund_id.name.apply(lambda s:s.find('沪深300'))
    # 采集基金净值数据，数据为
    value_start = '2015-01-01'
    fund_id_lst = fund_id.main_code.tolist()
    df = fund_value(value_start, fund_id_lst[0])
    for i in range(len(fund_id_lst))[1:]:
        print(fund_id_lst[i])
        df = pd.concat([df, fund_value(value_start, fund_id_lst[i])])
    df.to_csv(fold + 'fund_value_zf.csv')
    # df = pd.read_csv(fold + 'fund_value_zf.csv', index_col=0, encoding='gbk')
    # 计算每只基金评价指标
    #      fund_lst=fund_id.main_code.drop_duplicates().tolist()
    # 业绩基准序列
    d = df.day.drop_duplicates().tolist()
    d = [str(i) for i in d]
    d.sort(reverse=True)
    s = d[-1]
    e = d[0]
    benchmark = stock_price('000300.XSHG', '1d', s, e)
    benchmarknet = benchmark.close.tolist()
    # 统计
    df = df.dropna()
    ret = list()
    for idx, group in df.groupby('code'):
        if idx == 7490:
            print(idx)
        try:
            res = list()
            net = group.sum_value.tolist()
            annROR_ = annROR(net, len(net))
            over_annROR_ = over_annROR(benchmarknet, net)
            maxRetrace_ = maxRetrace(net)
            avgRetrace_ = avgRetrace(net)
            Var_ = Var(net)
            yearsharpRatio_ = yearsharpRatio(net)
            SortinoRatio_ = SortinoRatio(net)
            IR_ = IR(net, len(net))
            res.append(idx)
            res.append(annROR_)
            res.append(over_annROR_)
            res.append(maxRetrace_)
            res.append(avgRetrace_)
            res.append(Var_)
            res.append(yearsharpRatio_)
            res.append(SortinoRatio_)
            res.append(IR_)
            ret.append(res)
        except:
            res.append(idx)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
            ret.append(res)

    fund_ret = pd.DataFrame(ret, columns=['code', 'annROR', 'over_annROR', 'maxRetrace', 'avgRetrace', 'Var',
                                          'yearsharpRatio', 'SortinoRatio', 'IR'])
    # fund_ret.code = fund_ret.code.apply(lambda s: int(s))
    fund_ret = fund_ret.dropna()
    id_name = fund_id[['main_code', 'name', 'start_date', 'underlying_asset_type_id']]
    id_name.columns = ['code', 'name', 'start_date', 'type']
    # id_name.code = id_name.code.apply(lambda s: int(s))
    fund_ret_df = id_name.merge(fund_ret, on=['code'])

    fund_ret = fund_ret_df.copy()

    # cov_fund_factors=fund_ret.iloc[:,3:].corr()
    # cov_fund_factors.columns=['annROR','over_annROR','maxRetrace','avgRetrace','Var','yearsharpRatio','SortinoRatio','IR']
    # cov_fund_factors.index=['annROR','over_annROR','maxRetrace','avgRetrace','Var','yearsharpRatio','SortinoRatio','IR']
    # cov_fund_factors.to_excel('result/cov_factors_zf_'+str(i)+'.xls')
    # 基金筛选
    fund_ret_get = fund_ret.query("annROR < 5")
    fund_ret_get = fund_ret_get[['code', 'name', 'start_date', 'type', 'annROR', 'avgRetrace', 'Var', 'IR']]
    zb_list = ['annROR', 'avgRetrace', 'Var', 'IR']
    direc = [1, -1, 1, 1]
    fund_ret_get = topsis(fund_ret_get, zb_list, direc, w=[1, 1, 1, 1])

    fund_ret_get = fund_ret_get.sort_values(by='topsis', ascending=False)
    fund_ret_get.code = fund_ret_get.code.apply(lambda s: str(s).zfill(6))
    fund_best = fund_ret_get.head(10)
    fund_ret_get.to_excel(fold + 'fund_mark_all_topsis_zf' + '.xls', encoding='gbk')

# =============================================================================
# test      
# ============================================================================
