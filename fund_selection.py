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


def transfer_code(x):
    x = str(int(x))
    if len(x) < 6:
        x = '0' * (6-len(x)) + x
    return x


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold = 'e:/fof/cgo/'
    fold_data = 'e:/fof/data/'
    df1 = pd.read_excel('g:/lfp/fund_abilitity.xlsx', encoding='gbk')[['code', '基金名称', '分位点']]
    df2 = pd.read_excel('g:/lfp/混合型基金评级结果.xls', encoding='gbk')[['code', 'topsis', 'EndDate']]
    df = df1.merge(df2, on=['code']).dropna().assign(code=lambda df: df.code.apply(lambda x: transfer_code(x)))
    df.to_excel('g:/lfp/select_zf.xls', encoding='gbk')