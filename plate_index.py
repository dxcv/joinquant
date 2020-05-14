# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:26:59 2020
计算生成板块指数
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
auth('18610039264', 'zg19491001')


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
    start_day = '2014-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0.00

    lst = []
    df = pd.DataFrame([], columns=['trade_date'])
    for i in range(len(indus_name_lst)):
        indus_name = indus_name_lst[i]
        # new_energe = normalize_code(pd.read_csv(fold + indus_name + '.csv').stockcode.tolist())
        # # 生成制定行业指数
        # new_energe_index = auto_generate_index(new_energe, start_day)
        # print(new_energe_index)
        # new_energe_index.to_csv(fold + 'index' + indus_name + '.csv', encoding='gbk')
        print(indus_name)
        index = stock_price_jz(indus_name, start_day, end_day)[['date', 'name', 'code', 'open', 'close']]\
            .assign(trade_date=lambda df: df.date.apply(lambda x: str(x)[:10]))
        index = index.dropna()
        print(index)
        # index = pd.read_csv(fold + 'index' + indus_name + '.csv', encoding='gbk', index_col=0)\
        #     [['tradedate', 'index_', 'chg']]\
        #     .rename(columns={'index_': 'idx'})
        # print(index)
        position = pd.read_csv(fold + 'res_' + indus_name[:6] + '.csv', encoding='gbk', index_col=0)[['ENDDATE', 'position']]\
            .assign(position=lambda df: df.position.shift(1))\
            .assign(trade_date=lambda df: df.ENDDATE.apply(lambda x: str(x)[:10]))\
            .merge(index, on=['trade_date'])\
            .assign(close_1=lambda df: df.close.shift(1)).dropna()
        print(position)
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
                    net = net * s_price/_row.close_1
                    pos = 0
                elif _row.position == 1:
                    net = net * _row.close / _row.close_1
            net_lst.append(net)
        position['net'] = net_lst
        position['close_net'] = position['close']/position['close'].tolist()[0]
        annR = annROR(net_lst, 1)
        maxDraw = maxRetrace(net_lst, 1)
        sharp = yearsharpRatio(net_lst, 1)
        row = []
        row.append(name_lst[i])
        row.append(indus_name)
        row.append(trd_time)
        row.append(position.position.sum())
        row.append(len(position) - position.position.sum())
        row.append(annR)
        row.append(maxDraw)
        row.append(sharp)
        lst.append(row)
        position.index = pd.to_datetime(position.trade_date)
        position.ix[:, ['net', 'close_net']].plot()
        plt.show()
        chg_df = position[['trade_date', 'net', 'close_net']]
        chg_df['chg' + str(i)] = chg_df['net']/chg_df['net'].shift(1) - 1
        chg_df['close_chg' + str(i)] = chg_df['close_net']/chg_df['close_net'].shift(1) - 1
        chg_df = chg_df.fillna(value=0).reset_index(drop=True)
        df = df.merge(chg_df[['trade_date', 'chg' + str(i), 'close_chg' + str(i)]], on=['trade_date'], how='outer')

    trd_state_df = pd.DataFrame(lst, columns=['indus_name', 'code', 'trd_time', 'hold_day', 'short_position', 'annR',
                                              'maxRetrace', 'sharp'])
    trd_state_df['ave_hold_day'] = trd_state_df['hold_day']/trd_state_df['trd_time']
    df = df.fillna(value=0)
    df['chg'] = (df.chg0 + df.chg1 + df.chg2 + df.chg3 + df.chg4)/5
    df['close_chg'] = (df.close_chg0 + df.close_chg1 + df.close_chg2 + df.close_chg3 + df.close_chg4) / 5


    df['net'] = (1 + df['chg']).cumprod()
    df['index'] = (1 + df['close_chg']).cumprod()
    # position = position[position['tradedate'] > '2017-01-01']
    df.index = pd.to_datetime(df.trade_date)
    df.ix[:, ['net', 'index']].plot()
    plt.show()


