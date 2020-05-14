# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pymongo
import datetime
import copy
import math
from arctic import Arctic, TICK_STORE, CHUNK_STORE
style.use('ggplot')


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
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold = 'e:/fof/cgo/'
    fold_data = 'e:/fof/data/'
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    start_day = '2010-01-01'
    fee = 0.00
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    name_lst = ['bdt', 'xnyc', 'hlw', 'gkj', 'rgzn']
    indus_name_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    hq_dict = {}
    for i in range(len(indus_name_lst)):
        indus_name = indus_name_lst[i]
        name = name_lst[i]
        hq_df = stock_price_jz(indus_name, start_day, end_day)\
            .assign(date_time=lambda df: df.date.apply(lambda x: str(x)[:10])).dropna()
        # hq_df.to_csv(fold_data + 'hq_' + name + '.csv', encoding='gbk')
        hq_dict[name] = hq_df
    position_all = pd.read_csv(fold + 'indus_pos_df_all.csv', encoding='gbk')\
        .assign(date_time=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))\
        .assign(position=lambda df: df.total.shift(1))[
        ['date_time', 'position']]
    position_all = position_all[position_all['date_time'] < '2020-02-01']

    for i in range(len(indus_name_lst)):
        indus_name = indus_name_lst[i]
        name = name_lst[i]
        hq_df = hq_dict[name]
        position = position_all.merge(hq_df, on=['date_time'])\
            .assign(close_1=lambda df: df.close.shift(1)).dropna()

        pos = 0
        net_lst = []
        net = 1
        trd_time = 0
        for idx, _row in position.iterrows():
            if pos == 0:
                if _row.position > 0:
                    cost = _row.open * (1 + fee)
                    pos = _row.position
                    net = (pos * _row.close / cost + (1 - pos)) * net
                    trd_time += 1
                    pos = 1
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
                    net = net * (chg_pos * _row.close / cost + pos * _row.close / _row.close_1 + (1-_row.position))
                    pos = _row.position
                elif _row.position < pos:
                    chg_pos = pos - _row.position
                    s_price = _row.open * (1 - fee)
                    net = net * (chg_pos * s_price / _row.close_1 + _row.position * _row.close / _row.close_1 + (1-pos))
                    pos = _row.position
            net_lst.append(net)
        position['net'] = net_lst
        position['close_net'] = position['close'] / position['close'].tolist()[0]
        chg_df = position[['date_time', 'net', 'close_net']]
        chg_df['date_time'] = pd.to_datetime(chg_df['date_time'])
        chg_df = chg_df.set_index(['date_time'])
        sharpe_ratio = yearsharpRatio(net_lst, 1)
        sharpe = yearsharpRatio(position['close_net'].tolist(), 1)
        ann_return = annROR(net_lst, 1)
        max_drawdown = maxRetrace(net_lst, 1)
        title_str = 'sharpe %.2f %.2f ann_return %.2f max_drawdown %.2f' % (
            sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)
        chg_df.ix[:, ['net', 'close_net']].plot()
        plt.title(title_str)
        plt.show()

