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
import talib as tb

auth('18610039264', 'zg19491001')


# 获取价格
def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=False, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'})
    temp['stock_code'] = sec
    return temp

def trans_heng_float(x):
    if x == '--':
        x = None
    return x


def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.wind_index'].read(sec)
    temp['trade_date'] = temp.index
    temp = temp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    if sec == '930606.CSI':
        print(temp)
    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x)))\
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x)))\
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[['high', 'open', 'low', 'close', 'trade_date']]
    temp = temp.fillna(method='backfill', axis=1)

    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)]

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
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
    fold_data = 'e:/fof/data/'
    # indus_name_lst = ['tech']
    start_day = '2014-01-01'
    back_sdate = '2015-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0.0012
    N_ATR = 20
    ATR_n = 4
    index_code_lst = get_all_securities(types=['index'], date=end_day).index.tolist()
    etf_to_idx = pd.read_excel(fold_data + 'etf_to_idx_ymjh_indus.xls', encoding='gbk')[
        ['基金代码', '基金简称', '跟踪指数', '跟踪指数代码', 'select']] \
        .rename(columns={'基金代码': 'code', '基金简称': 'name', '跟踪指数代码': 'idx_code', '跟踪指数': 'idx_name'})
    etf_to_idx = etf_to_idx[etf_to_idx['select'] == 1]
    etf_to_idx = etf_to_idx.drop_duplicates(['idx_code'], keep='first')\
        .assign(idx_code=lambda df: df.idx_code.apply(lambda x: str(x))).sort_values(['idx_code'])
    etf_to_idx = etf_to_idx.assign(code_trans=lambda df: df.code.apply(lambda x: str(x)[:6])) \
        .assign(idx_code=lambda df: df.idx_code.apply(lambda x: str(x)[:6]))
    jz_idx_code_lst = jzmongo['stock_raw.wind_index'].list_symbols()
    jz_idx_code_df = pd.DataFrame(jz_idx_code_lst, columns=['jz_code'])
    jz_idx_code_df['temp'] = jz_idx_code_df.jz_code.apply(lambda x: x[0])
    jz_idx_code_df = jz_idx_code_df[(jz_idx_code_df['temp'] == '9') | (jz_idx_code_df['temp'] == '0') |
                                    (jz_idx_code_df['temp'] == 'H') | (jz_idx_code_df['temp'] == '3')]
    jz_idx_code_df = jz_idx_code_df.assign(idx_code=lambda df: df.jz_code.apply(lambda x: x[:6]))
    jz_idx_code_df = jz_idx_code_df.merge(etf_to_idx, on=['idx_code'])
    jz_idx_code_df.to_csv(fold_data + 'cgo_select_idx_code.csv', encoding='gbk')

    index_code_lst = jz_idx_code_df.jz_code.tolist()
    index_code_lst = list(set(index_code_lst))
    pos_df_all = pd.read_csv('e:/fof/cgo/' + 'indus_pos_df_all.csv', encoding='gbk')\
        .assign(position=lambda df: df.total.shift(1))[['trade_date', 'position']]\
        .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()

    index_hq_dict = {}
    for i in range(len(index_code_lst)):
        index_code = index_code_lst[i]
        index_hq = stock_price_jz(index_code, start_day, end_day)
        # index_hq = index_hq[(index_hq['trade_date'] >= start_day) & (index_hq['trade_date'] <= end_day) & (index_hq['money'] > 0)]
        index_hq_dict[index_code] = index_hq

    chg_df_all = pd.DataFrame(columns=['trade_date'])
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        # index_name = index_code_lst[j]
        index_name = jz_idx_code_df.set_index(['jz_code']).loc[index_code].idx_name

        pos_df = index_hq_dict[index_code].merge(pos_df_all, on=['trade_date']).sort_values(['trade_date']) \
            .assign(close_1=lambda df: df.close.shift(1)).dropna()

        # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # ax1 = ax.twinx()
        # pos_df.index = pd.to_datetime(pos_df['trade_date'])
        #
        # pos_df[['close']].plot(ax=ax1, figsize=(9, 6), kind='line', style=['k-'])
        # pos_df[['position']].plot(kind='area', grid=True, ax=ax, figsize=(9, 7), rot=60, style=['y'])
        #
        # title_str = '%s' % (index_name)
        # plt.savefig(fold + 'fig/' + title_str[:6] + '_position.png')
        # plt.show()
        position = pos_df.dropna()
        pos = 0
        net_lst = []
        net = 1
        trd_time = 0
        chg_pos = 0
        for idx, _row in position.iterrows():
            if pos == 0:
                if _row.position > 0:

                    cost = np.float(_row.open * (1 + fee))
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
                    net = net * (chg_pos * _row.close / cost + pos * _row.close / _row.close_1 + (1 - _row.position))
                    pos = _row.position
                elif _row.position < pos:
                    chg_pos = pos - _row.position
                    s_price = _row.open * (1 - fee)
                    net = net * (chg_pos * s_price / _row.close_1 + _row.position * _row.close / _row.close_1 + (
                            1 - pos))
                    pos = _row.position
            net_lst.append(net)
        position['net'] = net_lst
        position['close_net'] = position['close'] / position['close'].tolist()[0]
        chg_df = position[['trade_date', 'net', 'close_net']]
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
        title_str = '%s sharpe:%.2f idx_sharp:%.2f ann_return:%.2f max_drawdown:%.2f' % (
            index_name, sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)

        chg_df.ix[:, ['net', 'close_net']].plot()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title(title_str)
        plt.savefig(fold + 'fig/' + start_day + index_name + '.png')
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

    sharpe_ratio = yearsharpRatio(chg_df['net'].tolist(), 1)
    sharpe = yearsharpRatio(chg_df['close_net'].tolist(), 1)
    ann_return = annROR(chg_df['net'].tolist(), 1)
    max_drawdown = maxRetrace(chg_df['net'].tolist(), 1)
    title_str = 'profolio sharpe %.2f idx_sharp %.2f ann_return %.2f max_drawdown %.2f' % (
        sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)
    chg_df.ix[:, ['net', 'close_net']].plot()
    plt.title(title_str)
    plt.savefig(fold + 'fig/' + start_day + 'profolio.png')
    plt.show()




