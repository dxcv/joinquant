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


def get_prices(stock):
    all = pd.read_csv('E:/data/hq/' + 'pengbo.csv')
    try:
        tmp = all[['trade_date', stock]].rename(columns={stock: 'close'})
        tmp['trade_date'] = pd.to_datetime(tmp['trade_date'])
        tmp = tmp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).sort_values(['trade_date'])
    except:
        try:
            tmp = pd.read_csv('E:/data/hq/' + 'hq_1d' + '_' + stock + '_ly.csv') \
                .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).sort_values(
                ['trade_date'])
        except:
            tmp = pd.read_csv('E:/data/hq/' + 'hq_1d' + '_' + stock + '.csv') \
                .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).sort_values(['trade_date'])

    return tmp


def generate_stocks_price(stock_list):
    ret = []
    all = pd.read_csv('E:/data/hq/' + 'pengbo.csv')
    for i in stock_list:
        try:
            tmp = all[['trade_date', i]].rename(columns={i: 'close'})
            tmp['trade_date'] = pd.to_datetime(tmp['trade_date'])
            tmp = tmp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).sort_values(['trade_date'])
            tmp['chg'] = tmp.close.diff() / tmp.close
        except:
            tmp = pd.read_csv('E:/data/hq/' + 'hq_1d' + '_' + i + '.csv')\
                .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).sort_values(['trade_date'])
            tmp['chg'] = tmp.close.diff() / tmp.close
        ret.append(tmp)
    ret = pd.concat(ret)
    return ret


def generate_index(stocks_price):
    tmp = stocks_price[['trade_date', 'chg']].groupby('trade_date') \
        .mean().reset_index()
    tmp.chg.fillna(0, inplace=True)
    tmp['idx'] = (1 + tmp['chg']).cumprod()
    return tmp


def auto_generate_index(stock_list):
    tmp = generate_stocks_price(stock_list)
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
    fold = 'e:/future/data/'
    name_lst = ['tech', 'ai', 'chip', 'internet', 'energe']
    indus_name_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    # indus_name_lst = ['tech']
    start_day = '2014-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0.00
    class_lst = ['谷物', '化工', '有色金属', '黑色', '股指', '债券', '贵金属', '轻工业',
                 '原油', '油脂油料', '软商品', '农副产品']
    symbols_dict = {'股指': ['IF', 'IH', 'IC'],  # 股指
                    '债券': ['T', 'TF'],  # 债券
                    '贵金属': ['AG', 'AU'],
                    '谷物': ['C', 'CS', 'A', 'B'],  # 农产品
                    '化工': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA'],  # 化工
                    '有色金属': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],  # 金属
                    '黑色': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM'],  # 黑色系
                    '轻工业': ['SP', 'FG'],  #
                    '原油': ['SC', 'FU'],
                    '油脂油料': ['A', 'B', 'M', 'RM', 'Y', 'P', 'OI'],
                    '软商品': ['CF', 'CS', 'SR'],
                    '农副产品': ['JD', 'AP', 'CJ']
                    }

    df = pd.DataFrame([], columns=['trade_date'])
    s_date1 = '2003-01-01'
    e_date1 = '2004-01-01'
    s_date2 = '2004-01-01'
    e_date2 = '2007-01-01'
    percentile_now_sdate = '2015-01-01'

    lst = []
    for cls in class_lst:
        symbol_lst = symbols_dict[cls]

        for symbol in symbol_lst:
            print(symbol)

            idx_hq = get_prices(symbol)
            print(idx_hq)
            if idx_hq.trade_date.tolist()[0] > s_date1:
                continue
            print(idx_hq)
            idx_hq = idx_hq.set_index(['trade_date'])
            # row = []
            # row.append(idx_hq.trade_date.tolist()[0])
            # row.append(symbol)
            # lst.append(row)
            try:

                min_price1 = idx_hq.loc[s_date1: e_date1].close.min()
                min_idx1 = idx_hq.loc[s_date1: e_date1].close.idxmin()
                max_price1 = idx_hq.loc[s_date1: e_date1].close.max()
                max_idx1 = idx_hq.loc[s_date1: e_date1].close.idxmax()
                max_price2 = idx_hq.loc[s_date2: e_date2].close.max()
                max_idx2 = idx_hq.loc[s_date2: e_date2].close.idxmax()
                min_price2 = idx_hq.loc[s_date2: e_date2].close.min()
                min_idx2 = idx_hq.loc[s_date2: e_date2].close.idxmin()
                l_chg = max_price2 / min_price1 - 1
                l_days = (pd.to_datetime(max_idx2) - pd.to_datetime(min_idx1)).days
                s_chg = min_price2 / max_price1 - 1
                s_days = (pd.to_datetime(min_idx2) - pd.to_datetime(max_idx1)).days
                percentile_lsdate = str(int(min_idx1[:4]) - 5) + min_idx1[4:]
                percentile_ssdate = str(int(max_idx1[:4]) - 5) + max_idx1[4:]
                history_hq_long = idx_hq.loc[percentile_lsdate: min_idx1]
                history_hq_long['rank'] = history_hq_long.close.rank(method='min').astype(int)
                history_hq_long['percentile'] = history_hq_long['rank']/len(history_hq_long)
                history_hq_long_percentile = history_hq_long['percentile'].tolist()[-1]

                history_hq_shrot = idx_hq.loc[percentile_ssdate: max_idx1]
                history_hq_shrot['rank'] = history_hq_shrot.close.rank(method='min').astype(int)
                history_hq_shrot['percentile'] = history_hq_shrot['rank'] / len(history_hq_shrot)
                history_hq_short_percentile = history_hq_shrot['percentile'].tolist()[-1]

                history_hq_now = idx_hq.loc[percentile_now_sdate: end_day]
                history_hq_now['rank'] = history_hq_now.close.rank(method='min').astype(int)
                history_hq_now['percentile'] = history_hq_now['rank'] / len(history_hq_now)
                history_hq_now_percentile = history_hq_now['percentile'].tolist()[-1]

                row = []
                row.append(cls)
                row.append(symbol)
                row.append(min_price1)
                row.append(min_idx1)
                row.append(max_price1)
                row.append(max_idx1)
                row.append(max_price2)
                row.append(max_idx2)
                row.append(min_price2)
                row.append(min_idx2)
                row.append(l_chg)
                row.append(l_days)
                row.append(s_chg)
                row.append(s_days)
                row.append(history_hq_long_percentile)
                row.append(history_hq_short_percentile)
                row.append(history_hq_now_percentile)
                lst.append(row)
            except:
                continue

    df = pd.DataFrame(lst, columns=['商品类别', '品种', 'min_price_s', '起涨时间点', 'max_price_s', '起跌时间点', 'max_price_e',
                                    '多头结束时间点', 'min_price_e', '空头结束时间点', '上涨最大幅度', '上涨持续天数',
                                    '下跌最大幅度', '下跌持续天数', '历史起涨点分位点', '历史起跌点分位点', '当前分位点'])
    df_long = df[['商品类别', '品种', '起跌时间点', '空头结束时间点', '下跌最大幅度', '下跌持续天数', '历史起跌点分位点', '当前分位点']] \
        .sort_values(['下跌最大幅度'])
    # df_long = df[['商品类别', '品种', '起涨时间点', '多头结束时间点', '上涨最大幅度', '上涨持续天数', '历史起涨点分位点', '当前分位点']] \
    #     .sort_values(['上涨最大幅度'], ascending=False)
    # df_long = pd.read_csv(fold + 'symbol_chg' + s_date1 + '_' + s_date2 + '_' + e_date2 + 'long.csv', encoding='gbk')
    # df_long = df_long[['商品类别', '品种', '起跌时间点', '空头结束时间点', '下跌最大幅度', '下跌持续天数', '历史起跌点分位点', '当前分位点']] \
    #     .sort_values(['商品类别', '下跌最大幅度'])

    df_long['下跌最大幅度'] = -1 * df_long['下跌最大幅度']
    # df_long['下跌持续天数'] = np.abs(df_long['下跌持续天数'])
    df_long = df_long.set_index(['商品类别', '品种'])
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax1 = ax.twinx()

    df_long[['下跌持续天数']].plot(ax=ax1, kind='line', grid=True, figsize=(9, 6), rot=60, style=['r'])
    df_long[['下跌最大幅度']].plot(figsize=(9, 6), ax=ax, kind='bar')



    # df_long.ix[:, ['下跌最大幅度', '下跌持续天数']].plot(secondary_y=['下跌持续天数'])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('2003年各品种市场下行表现')
    plt.show()
    print(df)
    df_long.to_csv(fold + 'symbol_chg' + s_date1 + '_' + s_date2 + '_' + e_date2 + 'short.csv', encoding='gbk')
    print(df)
    # df.to_csv(fold + 'symbol_chg' + s_date1 + '_' + s_date2 + '_' + e_date2 + '.csv', encoding='gbk')





