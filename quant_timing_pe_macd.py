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
import talib
import math
from joinquant import *
from arctic import Arctic, TICK_STORE, CHUNK_STORE

style.use('ggplot')

def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=False, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'})\
        .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10]))
    temp = temp[temp['volume'] > 0]
    temp['stock_code'] = sec
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


def trans_heng_float(x):
    if x == '--':
        x = None
    return x


def index_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.wind_index'].read(sec)
    temp['trade_date'] = temp.index
    temp = temp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    if sec == '930606.CSI':
        print(temp)
    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x))) \
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x))) \
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[
        ['high', 'open', 'low', 'close', 'trade_date']]
    temp = temp.fillna(method='backfill', axis=1)

    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)]

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
    return temp


def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.stock_1d_jq_post'].read(sec)
    temp['trade_date'] = temp.index
    temp = temp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x))) \
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x))) \
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[
        ['high', 'open', 'low', 'close', 'trade_date']]
    temp = temp.fillna(method='backfill', axis=1)

    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].assign(stock_code=sec)

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
    return temp


def get_stock_pe(sec, sday, eday):
    temp = jzmongo['stock_raw.stock_valuation'].read(sec)
    temp = temp.assign(trade_date=lambda df: df.date.apply(lambda x: str(x)[:10]))[['trade_date', 'pe_ratio']]
    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].assign(stock_code=sec)
    return temp


def stock_pb(sec, sday, eday):
    temp = jzmongo['stock_raw.stock_valuation'].read(sec)\
        .assign(trade_date=lambda df: df.date.apply(lambda x: str(x)[:10]))[['trade_date', 'pb_ratio']]
    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].assign(stock_code=sec)
    return temp


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold = 'e:/fof/pe/'
    fold_data = 'e:/fof/data/'
    s_day = '2010-01-01'
    e_day = '2021-01-01'
    period = 756
    pecent_mid = 0.5
    pecent_up = 0.75
    pecent_down = 0.25
    weight_long = 0.2
    weight_short = 0.1
    width_long = 0.05
    width_short = 0.1
    index_code_lst = ['399006.XSHE', '000300.XSHG', '000905.XSHG', '000016.XSHG']
    code = index_code_lst[1]
    stocks_list = index_stocks(code)
    stock_code_lst = normalize_code(stocks_list)
    # stock_code_lst = ['000938.XSHE']
    for stock_code in stock_code_lst[:30]:
        try:
            print(stock_code)
            stock_hq = stock_price(stock_code, '1d', s_day, e_day)
            stock_pe = get_stock_pe(stock_code, s_day, e_day).reset_index(drop=True)
            lst_mid = []
            lst_up = []
            lst_down = []
            pe_ratio_lst = stock_pe.pe_ratio.tolist()
            for i in range(len(stock_pe)):
                if i < period:
                    lst_mid.append(None)
                    lst_up.append(None)
                    lst_down.append(None)
                else:
                    lst_mid.append(pd.Series(np.array(pe_ratio_lst[i-period: i])).quantile(pecent_mid))
                    lst_up.append(pd.Series(np.array(pe_ratio_lst[i-period: i])).quantile(pecent_up))
                    lst_down.append(pd.Series(np.array(pe_ratio_lst[i-period: i])).quantile(pecent_down))
            stock_pe['pe_mid'] = lst_mid
            stock_pe['pe_up'] = lst_up
            stock_pe['pe_down'] = lst_down
            stock_hq = stock_hq.merge(stock_pe, on=['trade_date']).sort_values(['trade_date'])\
                .assign(close_1=lambda df: df.close.shift(1))\
                .assign(pe_ratio=lambda df: df.pe_ratio.shift(1)).dropna()
            net = 1
            net_lst = []
            pos_lst = []
            pos = 0
            sell_method = 0
            sell_price = 100000000000
            for idx, _row in stock_hq.iterrows():
                if pos == 0:
                    if _row.pe_ratio < _row.pe_down:
                        cost = _row.open
                        pos = weight_long
                        net = (pos * _row.close / cost + (1 - pos)) * net
                        sell_method = 0
                elif (pos > 0) & (pos <= 1 - weight_long):
                    if (_row.low < cost * (1 - width_long)) & (_row.pe_ratio < _row.pe_mid):
                        cost = min(_row.open, cost * (1 - width_long))
                        chg_pos = weight_long
                        net = net * (chg_pos * _row.close / cost + pos * _row.close / _row.close_1 + (1 - pos - chg_pos))
                        pos = min(1, pos + chg_pos)
                        sell_method = 0
                    elif (sell_method == 0) & (_row.pe_ratio > _row.pe_up):
                        s_price = _row.open
                        chg_pos = weight_short
                        net = net * (chg_pos * s_price / _row.close_1 + (pos - chg_pos) * _row.close / _row.close_1 + (
                                1 - pos))
                        pos = max(0, pos - chg_pos)
                        sell_method = 1
                        sell_price = s_price
                    elif (sell_method == 1) & (_row.pe_ratio > _row.pe_mid) & (_row.high > sell_price * (1 + width_short)):
                        s_price = max(_row.open, sell_price * (1 + width_short))
                        chg_pos = width_short
                        net = net * (chg_pos * s_price / _row.close_1 + (pos - chg_pos) * _row.close / _row.close_1 + (
                                1 - pos))
                        pos = max(0, pos - chg_pos)
                        sell_method = 1
                        sell_price = s_price
                    else:
                        net = net * ((1 + pos) - pos * (2 - _row.close / _row.close_1))
                elif pos > 1-weight_long:
                    if (sell_method == 0) & (_row.pe_ratio > _row.pe_up):
                        s_price = _row.open
                        chg_pos = weight_short
                        net = net * (chg_pos * s_price / _row.close_1 + (pos - chg_pos) * _row.close / _row.close_1 + (
                                1 - pos))
                        pos = max(0, pos - chg_pos)
                        sell_method = 1
                        sell_price = s_price
                    elif (sell_method == 1) & (_row.pe_ratio > _row.pe_mid) & (_row.high > sell_price * (1 + width_short)):
                        s_price = max(_row.open, sell_price * (1 + width_short))
                        chg_pos = weight_short
                        net = net * (chg_pos * s_price / _row.close_1 + (pos - chg_pos) * _row.close / _row.close_1 + (
                                1 - pos))
                        pos = max(0, pos - chg_pos)
                        sell_method = 1
                        sell_price = s_price
                    else:
                        net = net * ((1 + pos) - pos * (2 - _row.close / _row.close_1))
                net_lst.append(net)
                pos_lst.append(pos)

            stock_hq['net'] = net_lst
            stock_hq['pos'] = pos_lst
            # stock_hq.to_csv(fold_data + 'stock_hq.csv')
            stock_hq.ix[:, ['pe_ratio', 'pe_mid', 'pe_up', 'pe_down']].plot()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.title('PE_BAND:' + stock_code[:-5])
            plt.savefig(fold + 'fig/' + stock_code[:-5] + 'PE_BAND' + '.png')
            plt.show()

            chg_df = stock_hq[['trade_date', 'net', 'close', 'pos']]
            chg_df['trade_date'] = pd.to_datetime(chg_df['trade_date'])
            chg_df = chg_df.set_index(['trade_date'])
            chg_df['net'] = chg_df['net'] / chg_df['net'].tolist()[0]
            chg_df['close_net'] = chg_df['close'] / chg_df['close'].tolist()[0]
            net_lst = chg_df['net'].tolist()
            sharpe_ratio = yearsharpRatio(net_lst, 1)
            sharpe = yearsharpRatio(chg_df['close_net'].tolist(), 1)
            ann_return = annROR(net_lst, 1)
            max_drawdown = maxRetrace(net_lst, 1)
            title_str = '%s sharpe:%.2f idx_sharp:%.2f ann_return:%.2f' % (
                stock_code[:-5], sharpe_ratio, sharpe, 100 * ann_return)

            chg_df.ix[:, ['net', 'close_net']].plot()
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.title(title_str)
            plt.savefig(fold + 'fig/' + stock_code + 'net' + '.png')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(9, 6))
            ax1 = ax.twinx()

            chg_df[['close']].plot(ax=ax1, figsize=(9, 6), kind='line', style=['k-'])
            chg_df[['pos']].plot(kind='area', grid=True, ax=ax, figsize=(9, 7), rot=60, style=['y'])

            title_str = '%s' % (stock_code[:-5] + 'position')
            plt.title(title_str)
            plt.savefig(fold + 'fig/' + title_str + '.png')
            plt.show()
        except Exception as e:
            print(str(e))


