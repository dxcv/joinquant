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
    fold_data = 'e:/future/data/'
    # myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    # jzmongo = Arctic(myclient)
    # start_day = '2010-01-01'
    # fee = 0.00
    # end_day = datetime.date.today().strftime('%Y-%m-%d')
    # name_lst = ['bdt', 'xnyc', 'hlw', 'gkj', 'rgzn']
    # indus_name_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    # hq_dict = {}
    # df1 = pd.read_csv(fold_data + 'WTI.csv')
    # df2 = pd.read_csv(fold_data + 'brent.csv')
    # df = df1.merge(df2, on=['trade_date']).assign(diff=lambda df: df.brent-df.WTI)
    # df = df[df['trade_date']>'2000-01-01']
    # df['trade_date'] = pd.to_datetime(df['trade_date'])
    # df = df.set_index(['trade_date'])
    # df.ix[:, ['WTI', 'brent', 'diff']].plot(secondary_y=['diff'])
    #
    # plt.show()

    mongoclient = pymongo.MongoClient("mongodb://192.168.2.51:27017/")
    store = Arctic(mongoclient)
    lib = store['foreign_market']
    item = lib.read('SM')
    df = item.data.rename(columns={'date': 'trade_date'})
    print(df)
    df.to_csv('E:/data/hq/' + 'hq_1d' + '_' + 'Y' + '_ly.csv')

