# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:26:59 2020
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
from matplotlib import style
# style.use('ggplot')

def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['wind_index'].read(sec)
    temp = temp[(temp['date'] >= sday) & (temp['date'] <= eday)]
    return temp


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
    fold = 'e:/fof/ymjh/'
    name_lst = ['bdt', 'xnyc', 'hlw', 'gkj', 'rgzn']
    index_code_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    # indus_name_lst = ['tech']
    start_day = '2010-01-01'
    back_sdate = '2010-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0.002
    # index_code_lst = ['399006.XSHE', '000300.XSHG', '000905.XSHG', '000016.XSHG']
    # name_lst = ['cyb', 'hs300', 'zz500', 'sz50']
    # index_code_lst = ['399006.XSHE', '000300.XSHG', '000016.XSHG']
    # name_lst = ['cyb', 'hs300', 'sz50']
    time_lst = [('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    para_dict = {'bdt': [(6, 16), (8, 18), (6, 24), (8, 14), (8, 22)],
                 'xnyc': [(6, 22), (6, 18), (8, 26), (6, 14), (4, 14)],
                 'hlw': [(6, 34), (22, 34)],
                 'gkj': [(6, 20), (12, 18), (12, 26), (18, 30), (6, 30)],
                 'rgzn': [(16, 22), (10, 26), (10, 26), (8, 26)]}
    index_hq_dict = {}
    pos_df_dict = {}
    for i in range(len(index_code_lst)):
        index_code = index_code_lst[i]
        index_hq = stock_price_jz(index_code, start_day, end_day) \
            .assign(trade_date=lambda df: df.date.apply(lambda x: str(x)[:10])).dropna()
        # index_hq.to_csv(fold + 'data/' + 'index_hq_' + index_code[:6] + '.csv', encoding='gbk')
        # index_hq = pd.read_csv(fold + 'data/' + 'index_hq_' + index_code[:6] + '.csv', encoding='gbk', index_col=0) \
        #     .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))
        # index_hq = index_hq[(index_hq['trade_date'] >= start_day) & (index_hq['trade_date'] <= end_day)]
        index_hq_dict[index_code] = index_hq
        money_position = pd.read_csv(fold + 'data/position_money/' + 'pos_large_money_' + index_code + '_p80_s5.csv')\
            .rename(columns={'position': 'pos_money'})
        money_position = money_position.assign(pos_money=lambda df: df.pos_money.shift(1)).fillna(0)
        # ymjh_position = pd.read_csv(fold + 'data/' + 'pos_' + index_code[:6] + '.csv').rename(columns={'position': 'pos_ymjh'})
        ymjh_position = pd.read_csv(fold + 'data/' + 'pos_' + index_code[:6] + '.csv')

        pos_df = money_position.merge(ymjh_position, on=['trade_date']).set_index(['trade_date'])
        pos_df['position_all'] = pos_df.min(axis=1)
        # pos_df['position'] = pos_df.min(axis=1)
        pos_df_dict[index_code] = pos_df

    lst = []
    portfolio_lst = []
    pos_df = pd.DataFrame(columns=['trade_date'])
    chg_df_all = pd.DataFrame(columns=['trade_date'])
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        index_name = name_lst[j]
        index_hq = index_hq_dict[index_code]
        pos_all_df = pos_df_dict[index_code]

        pos_all_df = pos_all_df.reset_index(drop=False).assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))

        pos_all_df.to_csv(fold + 'data/pos_money_ymjh_' + index_code[:6] + '.csv')
        pos_all_df = pos_all_df.merge(index_hq, on=['trade_date']).sort_values(['trade_date'])\
            .assign(close_1=lambda df: df.close.shift(1)).dropna()
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        ax1 = ax.twinx()
        pos_all_df.index = pd.to_datetime(pos_all_df['trade_date'])

        pos_all_df[['close']].plot(ax=ax1, figsize=(9, 6), kind='line', style=['k-'])
        pos_all_df[['position']].plot(kind='area', grid=True, ax=ax, figsize=(9, 7), rot=60, style=['y'])

        title_str = '%s' % (index_name)

        # pos_all_df.ix[:, ['position', 'close']].plot(kind='line', grid='on', secondary_y='position',
        #                                                      title=title_str + '_position')

        plt.savefig(fold + 'fig/' + title_str + '_position.png')
        plt.show()

        pos = 0
        net_lst = []
        net = 1
        trd_time = 0
        for idx, _row in pos_all_df.iterrows():
            if pos == 0:
                if _row.position > 0:
                    cost = _row.open * (1 + fee)
                    pos = _row.position
                    net = (pos * _row.close / cost + (1 - pos)) * net
                    trd_time += 1
                    pos = _row.position
            elif pos > 0:
                if _row.position == 0:
                    s_price = _row.open * (1 - fee)
                    net = net * (pos * s_price / _row.close_1 + (1 - pos))
                    pos = 0
                elif _row.position == pos:
                    pos = pos
                    net = net * ((1 + pos) - pos * (2 - _row.close / _row.close_1))
                elif _row.position > pos:
                    chg_pos = _row.position - pos
                    cost = _row.open * (1 + fee)
                    net = net * (chg_pos * _row.close / cost + pos * _row.close / _row.close_1 + (1 - _row.position))
                    pos = _row.position
                elif _row.position < pos:
                    chg_pos = pos - _row.position
                    s_price = _row.open * (1 - fee)
                    net = net * (chg_pos * s_price / _row.close_1 + _row.position * _row.close / _row.close_1 + (
                                1 - pos))
                    pos = _row.position
            net_lst.append(net)
        pos_all_df['net'] = net_lst
        pos_all_df['close_net'] = pos_all_df['close'] / pos_all_df['close'].tolist()[0]
        chg_df = pos_all_df[['trade_date', 'net', 'close_net']]
        chg_df = chg_df[chg_df['trade_date'] > back_sdate]
        chg_df['trade_date'] = pd.to_datetime(chg_df['trade_date'])
        chg_df = chg_df.set_index(['trade_date'])
        chg_df['net'] = chg_df['net'] / chg_df['net'].tolist()[0]
        chg_df['close_net'] = chg_df['close_net'] / chg_df['close_net'].tolist()[0]
        net_lst = chg_df['net'].tolist()
        sharpe_ratio = yearsharpRatio(net_lst, 1)
        sharpe = yearsharpRatio(chg_df['close_net'].tolist(), 1)
        ann_return = annROR(net_lst, 1)
        max_drawdown = maxRetrace(net_lst, 1)
        title_str = '%s sharpe:%.2f idx_sharp:%.2f ann_return:%.2f max_drawdown:%.2f' % (index_name,
            sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)

        chg_df.ix[:, ['net', 'close_net']].plot()
        plt.title(title_str)
        plt.savefig(fold + 'fig/' + index_name + '_profolio.png')
        plt.show()
        chg_df['chg' + str(j)] = chg_df['net'] / chg_df['net'].shift(1) - 1
        chg_df['close_chg' + str(j)] = chg_df['close_net'] / chg_df['close_net'].shift(1) - 1

        chg_df = chg_df.fillna(value=0).reset_index(drop=False)[
            ['trade_date', 'chg' + str(j), 'close_chg' + str(j)]]
        chg_df_all = chg_df_all.merge(chg_df, on=['trade_date'], how='outer')
    chg_df_all = chg_df_all.fillna(value=0)
    chg_df = chg_df_all.sort_values(['trade_date']).set_index(['trade_date'])
    chg_name = ['chg' + str(m) for m in range(len(index_code_lst))]
    close_chg_name = ['close_chg' + str(m) for m in range(len(index_code_lst))]
    chg_df['chg'] = chg_df[chg_name].sum(axis=1) / len(index_code_lst)
    chg_df['close_chg'] = chg_df[close_chg_name].sum(axis=1) / len(index_code_lst)
    chg_df['excess_chg'] = chg_df['chg'] - chg_df['close_chg']
    chg_df['net'] = (1 + chg_df['chg']).cumprod()
    chg_df['close_net'] = (1 + chg_df['close_chg']).cumprod()
    chg_df['net'] = chg_df['net'] / chg_df['net'].tolist()[0]
    chg_df['close_net'] = chg_df['close_net'] / chg_df['close_net'].tolist()[0]
    chg_df['excess_net'] = (1 + chg_df['excess_chg']).cumprod()
    chg_df = chg_df.reset_index(drop=False)
    chg_df['trade_date'] = pd.to_datetime(chg_df['trade_date'])
    chg_df = chg_df.set_index(['trade_date'])
    chg_df.ix[:, ['net', 'close_net']].plot()
    sharpe_ratio = yearsharpRatio(chg_df['net'].tolist(), 1)
    sharpe = yearsharpRatio(chg_df['close_net'].tolist(), 1)
    ann_return = annROR(chg_df['net'].tolist(), 1)
    max_drawdown = maxRetrace(chg_df['net'].tolist(), 1)
    title_str = 'profolio sharpe %.2f idx_sharp %.2f ann_return %.2f max_drawdown %.2f' % (
        sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)
    plt.title(title_str)
    plt.savefig(fold + 'fig/' + 'profolio.png')
    plt.show()
