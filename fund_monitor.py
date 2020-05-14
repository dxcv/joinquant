#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:23:29 2019
基金监控面板 进行基金组合的分析和统计
@author: yeecall
"""

from __future__ import division
import pandas as pd
import numpy as np
import os
import scipy.optimize as sco
py_path = r'E:/fof'
os.chdir(py_path)
import matplotlib.pyplot as plt
import math
import talib
import datetime
# from statsmodels import regression
# import statsmodels.api as sm
from jqdatasdk import *

auth('18610039264', 'zg19491001')


# 提取基金净值数据
def fund_value(start_day, code):
    q = query(finance.FUND_NET_VALUE.code,
              finance.FUND_NET_VALUE.day,
              finance.FUND_NET_VALUE.sum_value).filter(finance.FUND_NET_VALUE.code == code,
                                                       finance.FUND_NET_VALUE.day > start_day)
    df = finance.run_query(q)
    return (df)


# 价格转净值
def priceTovalue(df, w):
    f = df.iloc[0, :].tolist()
    for i in range(len(f)):
        df.iloc[:, i] = df.iloc[:, i] * w[i] / f[i]
    df['portfolio'] = df.sum(axis=1)
    return df


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


def yearsharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


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


def port_ret(weights):
    return np.sum(annual_mean_rets * weights)


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_rets, weights)))


# Function to be minimized
def min_func_sharpe_ratio(weights):
    return -port_ret(weights) / port_vol(weights)


def min_func_ann(weights):
    return -np.sum(annual_mean_rets * weights)


def min_func_momentum(weights):
    return -np.sum(momentum * weights)


def transfer_weight_max_min(x, weight_max, weight_min):
    if x < weight_min:
        x = weight_min
    elif x > weight_max:
        x = weight_max
    return x


def transfer_weight_adj_max_min(x, weight_max, weight_min, num_max, num_min, sum_exp):
    if (x > weight_min) & (x < weight_max):
        x = x * (1 - weight_max*num_max - weight_min*num_min) / sum_exp
    return x


def con(xmin, xmax, n):
    # 约束条件 分为eq 和ineq
    #eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0

    # cons = list()
    # for i in range(n):
    #     cons.append({'type': 'ineq', 'fun': lambda x: x[i] - xmin})
    #     cons.append({'type': 'ineq', 'fun': lambda x: -x[i] + xmax})

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[0] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[1] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[1] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[2] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[2] + xmax},\
            {'type': 'ineq', 'fun': lambda x: x[3] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[3] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[4] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[4] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[5] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[5] + xmax},\
            {'type': 'ineq', 'fun': lambda x: x[6] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[6] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[7] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[7] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[8] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[8] + xmax}, \
            {'type': 'ineq', 'fun': lambda x: x[9] - xmin}, \
            {'type': 'ineq', 'fun': lambda x: -x[9] + xmax}, \
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    return cons


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


def std_w(w, weight_max, weight_min):
    lst = []
    for i in w:
        if i > weight_max:
            lst.append(weight_max)
        elif i < weight_min:
            lst.append(weight_min)
        else:
            lst.append(i)
    ret = [i/sum(lst) for i in lst]
    return ret



if __name__ == '__main__':
    fund = ['512760.XSHG', '512480.XSHG', '515750.XSHG', '515000.XSHG', '515050.XSHG', '150246.XSHE', '515070.XSHG',
            '512930.XSHG', '515700.XSHG', '150212.XSHE']
    #      w=[1/len(fund) for i in fund] #权重
    #      w=opts['x']
    w = [0.18, 0.43, 0.06, 0.01, 0.31, 0.01]
    money = 1000000
    everyone = [money * i for i in w]
    flag = 1
    start = '2015-01-01'
    eday = '2020-02-20'
    weight_max = 0.3
    weight_min = 0.06
    time_lst = [('2019-10-01', '2020-12-31')]
    fold = 'e:/fof/'

    # 合成基金净值数据
    print("生成净值数据")
    print(fund[0])
    df = []
    for i in range(len(fund)):
        print(fund[i])
        temp = stock_price(fund[i], 'daily', start, eday)
        temp = temp.dropna()
        print(temp)
        df.append(temp)
    df = pd.concat(df)
    df = df.rename(columns={'tradedate': 'day', 'close': 'sum_value', 'stockcode': 'code'})
    # df.code = df.code.apply(lambda s: str(s) + '.jj')

    df = pd.pivot_table(df, index='day', columns='code', values='sum_value')
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    print(df)
    df.to_csv(fold + 'etf_sum_value_10.csv', encoding='gbk')
    # df = pd.read_csv(fold + 'etf_sum_value_10.csv', encoding='gbk', index_col=0)
    df = df.loc['2019-08-01':, :]
    df = df[fund]
    print(df)


    if flag == 0:
        # 生成净值
        ret = list()
        fundvalue = priceTovalue(df, w)
        fundvalue = fundvalue.dropna()

        fundvalue.loc[:, 'portfolio'].reset_index().plot()

        net = fundvalue.portfolio.tolist()
        net_252 = net[-252:]
        days = fundvalue.index.tolist()
        s = str(days[0])[:10]
        e = str(days[-1])[:10]
        # 近7天收益率：
        yoy_7 = (net[-1] - net[-7]) / net[-7]
        # 据最高净值回撤：
        H_Retrace = (net[-1] - max(net)) / max(net)
        # 近1年收益率：
        yoy_252 = (net[-1] - net[-252]) / net[-252]

        # 近一年最大回撤：
        maxRet_252 = maxRetrace(net_252)
        # 近一年夏普比率：
        sharp_252 = yearsharpRatio(net_252)
        # 近一年VAR:
        var_252 = Var(net_252)
        # 近一年最大回撤：
        maxRet = maxRetrace(net)
        # 近一年夏普比率：
        sharp = yearsharpRatio(net)
        # 近一年VAR:
        var = Var(net)

        ret.append(s)
        ret.append(e)
        ret.append(yoy_7)
        ret.append(H_Retrace)
        ret.append(yoy_252)
        ret.append(maxRet_252)
        ret.append(sharp_252)
        ret.append(var_252)
        ret.append(maxRet)
        ret.append(sharp)
        ret.append(var)
        result = pd.DataFrame(ret)
        result['说明'] = ['开始日期', '结束日期', '近七天收益率', '高水位回撤', '近1年收益率', '近一年最大回撤', '近一年夏普', '近一年VAR', '历史最大回撤', '历史夏普率',
                        'VAR']
        result.columns = ['值', '说明']
        print(result)
    # =============================================================================
    #  基于MTP 计算夏普最优参数
    # =============================================================================
    else:
        df = df.reset_index()
        df.day = df.day.apply(lambda s: str(s)[:10])
        w_lst = []
        # Calculate the annualized mean returns and covariance matrix
        for (s_date, e_date) in time_lst:
            df_ = df[(df['day'] >= s_date) & (df['day'] <= e_date)]
            #            df=df.query("day<'{var}'".format(var=end))
            df_ = df_.set_index('day', drop=True)
            col_name = df_.columns.values
            rets = np.log(df_ / df_.shift(1)).dropna()
            annual_mean_rets = rets.mean() * 252
            # The covrance of random walk is in proportion to time
            annual_cov_rets = rets.cov() * 252
            momentum = df_ - df_.shift(12)
            momentum = momentum.rolling(6).mean().iloc[-1]
            momentum_std = momentum/momentum.sum()
            w = momentum_std.values
            w = std_w(w, weight_max, weight_min)
            w_lst.append(w)
            annual_mean_rets_std = annual_mean_rets / annual_mean_rets.sum()
            w = annual_mean_rets_std.values
            w = std_w(w, weight_max, weight_min)
            w_lst.append(w)

            noa = len(df_.columns)
            # Equality constraint
            # cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            cons = con(weight_min, weight_max, len(fund))

            # Bounds for the parameters
            bnds = tuple((0, 1) for x in range(noa))

            # Initial parameters
            eweights = np.array(noa * [1. / noa, ])

            opts = sco.minimize(min_func_sharpe_ratio, eweights, method='SLSQP', bounds=bnds, constraints=cons)
            w = opts['x']
            w_lst.append(w)
            print(opts['x'])

            # df_.corr().to_csv('result/value_corr.csv')
        temp = pd.DataFrame(w_lst, columns=col_name)
        temp = temp.T
        temp.columns = ['momentum', 'ann', 'sharp']
        temp['ave'] = temp.sum(axis=1) / len(temp.columns)

        temp.to_csv(fold + 'weight6.csv')
        temp['w0'] = temp['ave'].apply(lambda x: transfer_weight_max_min(x, weight_max, weight_min))

        num_min = len([i for i in temp['w0'] if i == weight_min])
        num_max = len([i for i in temp['w0'] if i == weight_max])
        sum_exp = sum([i for i in temp.w0 if i not in [weight_max, weight_min]])
        temp['w1'] = temp['w0'].apply(
            lambda x: transfer_weight_adj_max_min(x, weight_max, weight_min, num_max, num_min, sum_exp))
        score = pd.read_csv(fold + 'score_manager.csv')
        # score['code'] = score['code'].apply(lambda x: x[5:] + '.jj')

        temp['code'] = temp.index
        temp = temp.merge(score, on=['code'])
        temp['score'] = len(temp['基金经理评级']) * temp['基金经理评级'] / temp['基金经理评级'].sum()
        temp['weight_adj'] = temp['score'] * temp['w1']
        temp['weight_final'] = temp['weight_adj']/temp['weight_adj'].sum()
        temp.to_csv(fold + 'weight_final.csv', encoding='gbk')