# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:26:59 2020
仓位为 signal * cgo>0的成分股占比
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from dateutil.parser import parse
from jqdatasdk import *
import copy
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
import talib as tb

auth('18610039264', 'zg19491001')


# 获取价格
def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=['open', 'close', 'high', 'low'],
                     skip_paused=False, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'})
    temp['stock_code'] = sec
    return temp


def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['wind_index'].read(sec)
    temp = temp[(temp['date'] >= sday) & (temp['date'] <= eday)]
    return temp


# 提取成分股列表
def constituent_stock(df):
    df.stockcode = df.stockcode.apply(lambda s: normalize_code(s))
    return df.stockcode.drop_duplicates().tolist()


def get_prices(stock, s_t, e_t):
    return stock_price(stock, 'daily', s_t, e_t)


def generate_stocks_price(stock_list, s_t):
    e_t = str(datetime.datetime.today())[:10]
    ret = []
    for i in stock_list:
        tmp = get_prices(i, s_t, e_t)
        tmp['chg'] = tmp.close.diff() / tmp.close
        ret.append(tmp)
    ret = pd.concat(ret)
    return ret


def generate_index(stocks_price):
    tmp = stocks_price[['tradedate', 'chg']].groupby('tradedate') \
        .mean().reset_index()
    tmp.chg.fillna(0, inplace=True)
    tmp['index_'] = (1 + tmp['chg']).cumprod()
    return tmp


def auto_generate_index(stock_list, s_t):
    tmp = generate_stocks_price(stock_list, s_t)
    ret = generate_index(tmp)
    return ret


def yearsharpRatio(netlist, n):
    '''
    :param netlist:
    :param n: 每交易日对应周期数
    :return:
    '''
    row = []
    new_lst = copy.deepcopy(netlist)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]
    for i in range(1, len(new_lst)):
        row.append(math.log(new_lst[i] / new_lst[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


def maxRetrace(lst, n):
    '''
    :param list:netlist
    :param n:每交易日对应周期数
    :return: 最大历史回撤
    '''
    Max = 0
    new_lst = copy.deepcopy(lst)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]

    for i in range(len(new_lst)):
        if 1 - new_lst[i] / max(new_lst[:i + 1]) > Max:
            Max = 1 - new_lst[i] / max(new_lst[:i + 1])
    return Max


def annROR(netlist, n):
    '''
    :param netlist:净值曲线
    :param n:每交易日对应周期数
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 252 * n / len(netlist)) - 1


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    fold = 'e:/fof/cgo/'
    name_lst = ['tech', 'ai', 'chip', 'internet', 'energe']
    indus_name_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    # indus_name_lst = ['tech']
    start_day = '2010-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0.0005
    index_code_lst = ['399006.XSHE', '000300.XSHG', '000905.XSHG']
    name_lst = ['cyb', 'hs300', 'zz500']
    time_lst = [('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    para_dict = {'cyb': [(0.7, 8, 18), (0.9, 9, 18), (0.85, 11, 30), (0.85, 20, 38), (0.95, 15, 18)],
                 'hs300': [(0.95, 9, 22), (0.95, 12, 18), (0.9, 7, 54), (0.9, 14, 38), (0.9, 8, 12)],
                 'zz500': [(0.95, 9, 68), (0.65, 15, 38), (0.95, 14, 62), (0.95, 11, 22), (0.9, 13, 20)], }
    # para_dict = {'cyb': [(0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15)],
    #              'hs300': [(0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15)],
    #              'zz500': [(0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15), (0.3, 5, 15)],}

    # name_lst = ['cyb', 'hs300']
    s_period_lst = [i for i in range(4, 24, 1)]
    l_period_lst = [i for i in range(10, 91, 2)]
    pecent_lst = [i / 100 for i in range(65, 96, 5)]
    # s_period_lst = [i for i in range(5, 6)]
    # l_period_lst = [i for i in range(15, 16)]
    # pecent_lst = [i / 10 for i in range(3, 10, 10)]

    index_hq_dict = {}
    for i in range(len(index_code_lst)):
        index_code = index_code_lst[i]
        # index_hq = stock_price(index_code, '1d', start_day, end_day) \
        #     .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
        # index_hq.to_csv(fold + 'index_hq_' + index_code[:6] + '.csv', encoding='gbk')
        index_hq = pd.read_csv(fold + 'index_hq_' + index_code[:6] + '.csv', encoding='gbk', index_col=0) \
            .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))
        index_hq = index_hq[(index_hq['trade_date'] >= start_day) & (index_hq['trade_date'] <= end_day)]
        index_hq_dict[index_code] = index_hq
    cgo_dict = {}
    for i in range(len(index_code_lst)):
        index_code = index_code_lst[i]
        index_hq = pd.read_csv(fold + 'stock_cgo_' + index_code[:6] + '.csv', encoding='gbk', index_col=0)[
            ['ENDDATE', 'CGO']] \
            .assign(trade_date=lambda df: df.ENDDATE.apply(lambda x: str(x)[:10]))[['trade_date', 'CGO']]
        index_hq = index_hq[(index_hq['trade_date'] >= start_day) & (index_hq['trade_date'] <= end_day)]
        cgo_dict[index_code] = index_hq
    lst = []
    portfolio_lst = []
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        index_name = name_lst[j]
        index_hq = index_hq_dict[index_code]
        cgo_ori = cgo_dict[index_code]
        para_lst = para_dict[index_name]
        chg_df = pd.DataFrame(columns=['trade_date'])
        pos_df = pd.DataFrame(columns=['trade_date'])
        for i in range(len(para_lst)):
            para = para_lst[i]
            pecent = para[0]
            s_period = para[1]
            l_period = para[2]
            f = lambda s: s.quantile(pecent)

            print(pecent, s_period, l_period)
            cgo = cgo_ori.groupby('trade_date').CGO.apply(f).reset_index()

            position = cgo.assign(cgo_ma_l=lambda df: tb.MA(df.CGO.values, l_period)) \
                .assign(cgo_ma_s=lambda df: tb.MA(df.CGO.values, s_period)) \
                .assign(position=lambda df: df.cgo_ma_s > df.cgo_ma_l) \
                .assign(position=lambda df: df.position.apply(lambda x: int(x)))[['trade_date', 'position']] \
                .assign(position=lambda df: df.position.shift(1)) \
                .merge(index_hq, on=['trade_date']) \
                .assign(close_1=lambda df: df.close.shift(1)).dropna()
            trd_time = 0
            pos = 0
            net_lst = []
            net = 1
            for idx, _row in position.iterrows():
                if pos == 0:
                    if _row.position == 1:
                        cost = _row.open * (1 + fee)
                        net = net * _row.close / cost
                        trd_time += 1
                        pos = 1
                elif pos == 1:
                    if _row.position == 0:
                        s_price = _row.open * (1 - fee)
                        net = net * s_price / _row.close_1
                        pos = 0
                    elif _row.position == 1:
                        net = net * _row.close / _row.close_1
                net_lst.append(net)
            position['net'] = net_lst
            position['close_net'] = position['close'] / position['close'].tolist()[0]
            chg_df_ = position[['trade_date', 'net', 'close_net']]
            chg_df_['chg' + str(i)] = chg_df_['net'] / chg_df_['net'].shift(1) - 1
            chg_df_['close_chg' + str(i)] = chg_df_['close_net'] / chg_df_['close_net'].shift(1) - 1

            chg_df_ = chg_df_.fillna(value=0).reset_index(drop=True)[
                ['trade_date', 'chg' + str(i), 'close_chg' + str(i)]]
            chg_df = chg_df.merge(chg_df_, on=['trade_date'], how='outer')
        chg_df = chg_df.fillna(value=0)
        chg_df['chg'] = (chg_df['chg0'] + chg_df['chg2'] + chg_df['chg3'] + chg_df['chg4'] + chg_df['chg1'])/5
        chg_df['close_chg'] = chg_df['close_chg0']
        chg_df['excess_chg'] = chg_df['chg'] - chg_df['close_chg']
        chg_df['net'] = (1 + chg_df['chg']).cumprod()
        chg_df['idx'] = (1 + chg_df['close_chg']).cumprod()
        chg_df['excess_net'] = (1 + chg_df['excess_chg']).cumprod()
        chg_df['trade_date'] = pd.to_datetime(chg_df['trade_date'])
        chg_df = chg_df.set_index(['trade_date'])
        chg_df.ix[:, ['net', 'idx', 'excess_net']].plot()
        plt.title(index_name + '_lfp')
        plt.show()
    #         for (s_date, e_date) in time_lst:
    #             chg_df__ = chg_df_[(chg_df_['trade_date'] >= s_date) & (chg_df_['trade_date'] <= e_date)]
    #             chg_df__['excess_net_lst'] = (chg_df__['excess_chg' + str(i)] + 1).cumprod()
    #             excess_net_lst = chg_df__['excess_net_lst'].tolist()
    #             annR = annROR(excess_net_lst, 1)
    #             maxDraw = maxRetrace(excess_net_lst, 1)
    #             sharp = yearsharpRatio(excess_net_lst, 1)
    #             # print(annR)
    #             row = []
    #             row.append(name_lst[i])
    #             row.append(index_code)
    #             row.append(trd_time)
    #             row.append(position.position.sum())
    #             row.append(len(position) - position.position.sum())
    #             row.append(annR)
    #             row.append(maxDraw)
    #             row.append(sharp)
    #             row.append(s_period)
    #             row.append(l_period)
    #             row.append(pecent)
    #             row.append(s_date)
    #             row.append(e_date)
    #             lst.append(row)
    #             # position.index = pd.to_datetime(position.trade_date)
    #             # position.ix[:, ['net', 'close_net']].plot()
    #             # plt.show()
    #
    #             #     pos_df = pos_df.merge(position[['position', 'trade_date']].reset_index(drop=True).rename(
    #             #         columns={'position': 'position' + str(i)}), on=['trade_date'], how='outer')
    #             # pos_df = pos_df.dropna()
    #             # pos_df['position'] = pos_df.set_index(['trade_date']).sum(axis=1) / len(index_code_lst)
    #             # pos_df = pos_df[['trade_date', 'position']]
    #             # df = chg_df.fillna(value=0)
    #             # df['chg'] = (df.chg0 + df.chg1 + df.chg2) / 3
    #             # df['close_chg'] = (df.close_chg0 + df.close_chg1 + df.close_chg2) / 3
    #             # df['net'] = (1 + df['chg']).cumprod()
    #             # df['index'] = (1 + df['close_chg']).cumprod()
    #             # net_lst = df['net'].tolist()
    #             # net_idx_lst = df['index'].tolist()
    #             # annR = annROR(net_lst, 1)
    #             # maxDraw = maxRetrace(net_lst, 1)
    #             # sharp = yearsharpRatio(net_lst, 1)
    #             # annR_idx = annROR(net_idx_lst, 1)
    #             # maxDraw_idx = maxRetrace(net_idx_lst, 1)
    #             # sharp_idx = yearsharpRatio(net_idx_lst, 1)
    #             # row = []
    #             # row.append(annR)
    #             # row.append(maxDraw)
    #             # row.append(sharp)
    #             # row.append(annR_idx)
    #             # row.append(maxDraw_idx)
    #             # row.append(sharp_idx)
    #             # row.append(s_period)
    #             # row.append(l_period)
    #             # row.append(pecent)
    #             # portfolio_lst.append(row)
    #
    #             # df.index = pd.to_datetime(df.trade_date)
    #             # df.ix[:, ['net', 'index']].plot()
    #             # plt.show()
    # trd_state_df = pd.DataFrame(lst, columns=['indus_name', 'code', 'trd_time', 'hold_day', 'short_position', 'annR',
    #                                           'maxRetrace', 'sharp', 's_period', 'l_period', 'pecent', 's_date',
    #                                           'e_date'])
    # trd_state_df['ave_hold_day'] = trd_state_df['hold_day'] / trd_state_df['trd_time']
    # trd_state_df.to_csv(fold + 'trd_state_df1.csv', encoding='gbk')
    # # portfolio_df = pd.DataFrame(portfolio_lst, columns=['annR', 'maxRetrace', 'sharp', 'annR_idx', 'maxRetrace_idx',
    # #                                                     'sharp_idx', 's_period', 'l_period', 'pecent'])
    # #
    # # portfolio_df.to_csv(fold + 'portfolio_state_df.csv', encoding='gbk')
    #
    # # lst = []
    # # df = pd.DataFrame([], columns=['trade_date'])
    # # for i in range(len(indus_name_lst)):
    # #     indus_name = indus_name_lst[i]
    # #     # new_energe = normalize_code(pd.read_csv(fold + indus_name + '.csv').stockcode.tolist())
    # #     # # 生成制定行业指数
    # #     # new_energe_index = auto_generate_index(new_energe, start_day)
    # #     # print(new_energe_index)
    # #     # new_energe_index.to_csv(fold + 'index' + indus_name + '.csv', encoding='gbk')
    # #     print(indus_name)
    # #     index = stock_price_jz(indus_name, start_day, end_day)[['date', 'name', 'code', 'open', 'close']] \
    # #         .assign(trade_date=lambda df: df.date.apply(lambda x: str(x)[:10]))
    # #     index = index.dropna()
    # #     print(index)
    # #     position = pos_df[['trade_date', 'position']] \
    # #         .merge(index, on=['trade_date']) \
    # #         .assign(close_1=lambda df: df.close.shift(1)).dropna()
    # #     print(position)
    # #     trd_time = 0
    # #     pos = 0
    # #     net_lst = []
    # #     net = 1
    # #     for idx, _row in position.iterrows():
    # #         if pos == 0:
    # #             if _row.position > 0:
    # #                 pos = _row.position
    # #                 cost = _row.open * (1 + fee)
    # #                 net = net * (pos * _row.close / cost + (1 - pos))
    # #                 trd_time += 1
    # #         elif pos > 1:
    # #             if _row.position == 0:
    # #                 s_price = _row.open * (1 - fee)
    # #                 net = net * (pos * s_price / _row.close_1 + (1 - pos))
    # #                 pos = 0
    # #             elif _row.position > 0:
    # #                 net = net * (pos * _row.close / _row.close_1 + (1 - pos))
    # #         net_lst.append(net)
    # #     position['net'] = net_lst
    # #     position['close_net'] = position['close'] / position['close'].tolist()[0]
    # #     annR = annROR(net_lst, 1)
    # #     maxDraw = maxRetrace(net_lst, 1)
    # #     sharp = yearsharpRatio(net_lst, 1)
    # #     row = []
    # #     row.append(name_lst[i])
    # #     row.append(indus_name)
    # #     row.append(annR)
    # #     row.append(maxDraw)
    # #     row.append(sharp)
    # #     lst.append(row)
    # #     position.index = pd.to_datetime(position.trade_date)
    # #     position.ix[:, ['net', 'close_net']].plot()
    # #     plt.show()
    # #     chg_df = position[['trade_date', 'net', 'close_net']]
    # #     chg_df['chg' + str(i)] = chg_df['net'] / chg_df['net'].shift(1) - 1
    # #     chg_df['close_chg' + str(i)] = chg_df['close_net'] / chg_df['close_net'].shift(1) - 1
    # #     chg_df = chg_df.fillna(value=0).reset_index(drop=True)
    # #     df = df.merge(chg_df[['trade_date', 'chg' + str(i), 'close_chg' + str(i)]], on=['trade_date'], how='outer')
    # #
    # # trd_indus_state_df = pd.DataFrame(lst, columns=['indus_name', 'code', 'annR', 'maxRetrace', 'sharp'])
    # #
    # # df = df.fillna(value=0)
    # # df['chg'] = (df.chg0 + df.chg1 + df.chg2 + df.chg3 + df.chg4) / 5
    # # df['close_chg'] = (df.close_chg0 + df.close_chg1 + df.close_chg2 + df.close_chg3 + df.close_chg4) / 5
    # #
    # # df['net'] = (1 + df['chg']).cumprod()
    # # df['index'] = (1 + df['close_chg']).cumprod()
    # # # position = position[position['tradedate'] > '2017-01-01']
    # # df.index = pd.to_datetime(df.trade_date)
    # # df.ix[:, ['net', 'index']].plot()
    # # plt.show()
