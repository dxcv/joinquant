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
    fold = 'e:/fof/ymjh/'
    fold_data = 'e:/fof/data/'
    name_lst = ['tech', 'ai', 'chip', 'internet', 'energe']
    indus_name_lst = ['884160.WI', '884076.WI', '884224.WI', '399608.SZ', '884201.WI']
    # indus_name_lst = ['tech']
    start_day = '2014-01-01'
    back_sdate = '2015-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    fee = 0
    N_ATR = 20
    ATR_n = 4
    # index_code_lst = get_all_securities(types=['index'], date=end_day).index.tolist()
    # etf_to_idx = pd.read_excel(fold + 'data/etf_to_idx.xls', encoding='gbk')[
    #     ['基金代码', '基金简称', '跟踪指数', '跟踪指数代码', 'select']] \
    #     .rename(columns={'基金代码': 'code', '基金简称': 'name', '跟踪指数代码': 'idx_code', '跟踪指数': 'idx_name'})
    # etf_to_idx = etf_to_idx[etf_to_idx['select'] == 1]
    # etf_to_idx = etf_to_idx.assign(code_trans=lambda df: df.code.apply(lambda x: str(x)[:6])) \
    #     .assign(idx_code_trans=lambda df: df.idx_code.apply(lambda x: str(x)[:6]))
    # index_code_select_lst = [i for i in index_code_lst if str(i)[:6] in etf_to_idx.idx_code_trans.tolist()]
    # index_code_select_lst = list(set(index_code_select_lst))
    index_code_lst = ['000068.XSHG', '399364.XSHE', '000932.XSHG', '000989.XSHG', '399973.XSHE', '000928.XSHG', '000066.XSHG',
                '399550.XSHE', '399967.XSHE', '000063.XSHG', '000036.XSHG', '000048.XSHG', '000986.XSHG', '000987.XSHG',
                '000814.XSHG', '000056.XSHG']

    para_dict = {'000068': [(10, 26), (18, 54), (6, 66), (13, 28), (6, 26)],
                 '399364': [(8, 54), (28, 50), (10, 54), (6, 74), (16, 78)],
                 '000932': [(4, 42), (21, 34), (10, 62), (24, 38), (10, 54)],
                 '000989': [(7, 80), (9, 48), (5, 10), (13, 56)],
                 '399973': [(12, 26), (6, 22), (12, 18)],
                 '000928': [(24, 66), (34, 62), (10, 70), (8, 26), (10, 58)],
                 '000066': [(6, 50), (19, 56), (6, 58), (10, 30), (7, 48)],
                 '399550': [(26, 38), (10, 70), (6, 42), (16, 74)],
                 '399967': [(11, 36), (6, 26)],
                 '000063': [(21, 64), (27, 52), (7, 68), (7, 16), (7, 16)],
                 '000036': [(5, 38), (21, 28), (7, 64), (25, 40), (11, 48)],
                 '000048': [(17, 68), (5, 24), (5, 56), (7, 40), (6, 12)],
                 '000986': [(9, 72), (11, 38), (7, 18), (4, 18)],
                 '000987': [(4, 14), (9, 44), (6, 30), (27, 72)],
                 '000814': [(11, 16), (7, 48), (7, 48), (9, 28)],
                 '000056': [(21, 68), (25, 56), (7, 56), (17, 76), (8, 78)]
                 }
    index_hq_all = pd.read_csv(fold_data + 'index_hq_all.csv', encoding='gbk')

    index_hq_dict = {}
    for i in range(len(index_code_lst)):
        index_code = index_code_lst[i]
        index_hq = index_hq_all[index_hq_all['stock_code'] == index_code]\
            .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
        index_hq = index_hq[(index_hq['trade_date'] >= start_day) & (index_hq['trade_date'] <= end_day)]
        index_hq_dict[index_code] = index_hq
    lst = []
    portfolio_lst = []

    chg_df_all = pd.DataFrame(columns=['trade_date'])
    for j in range(len(index_code_lst)):
        pos_df = pd.DataFrame(columns=['trade_date'])
        index_code = index_code_lst[j]
        index_name = index_code_lst[j]
        index_hq = index_hq_dict[index_code]
        para_lst = para_dict[index_name[:6]]
        chg_df = pd.DataFrame(columns=['trade_date'])

        for i in range(len(para_lst)):
            para = para_lst[i]
            N1 = para[0]
            N2 = para[1]
            position = index_hq \
                .assign(HH_s=lambda df: tb.MAX(df.high.shift(1).values, N1)) \
                .assign(LL_s=lambda df: tb.MIN(df.low.shift(1).values, N1)) \
                .assign(HH_l=lambda df: tb.MAX(df.high.shift(1).values, N2)) \
                .assign(LL_l=lambda df: tb.MIN(df.low.shift(1).values, N2)) \
                .assign(ma_s=lambda df: (df.HH_s + df.LL_s) / 2) \
                .assign(ma_l=lambda df: (df.HH_l + df.LL_l) / 2) \
                .assign(ma_s1=lambda df: df.ma_s.shift(1)) \
                .assign(ma_l1=lambda df: df.ma_l.shift(1)) \
                .assign(ave_p=lambda df: (2 * df.close.shift(1) + df.high.shift(1) + df.low.shift(1)) / 4) \
                .assign(close_1=lambda df: df.close.shift(1))
            position['atr'] = tb.ATR(position['high'].values, position['low'].values,
                                         position['close'].values, N_ATR)
            position = position.assign(atr=lambda df: df.atr.shift(1))
            low_price_pre = 0
            high_price_pre = 100000000
            pos = 0
            pos_lst = []
            signal_lst = []
            trad_times = 0
            net = 1
            net_lst = []

            long_stop = False
            short_stop = False
            for idx, _row in position.iterrows():
                condition_l = ((_row.ma_s1 == _row.ma_l1) and (_row.ma_s > _row.ma_l) and (
                        _row.ave_p >= _row.ma_s)) or ((_row.ma_s1 < _row.ma_l1) and (
                        _row.ma_s > _row.ma_l) and (_row.ave_p >= min(_row.ma_s, _row.ma_l)))
                condition_s = (_row.ma_s1 > _row.ma_l1) and (_row.ma_s < _row.ma_l) and (
                        _row.ave_p <= max(_row.ma_s, _row.ma_l))
                if pos == 0:
                    if condition_l:
                        cost = _row.open * (1 + fee)
                        pos = 1
                        hold_price = []
                        high_price = []
                        high_price.append(_row.high)
                elif pos == 1:
                    if condition_s:
                        pos = 0
                    elif _row.close_1 < max(high_price) - _row.atr * ATR_n:
                        pos = 0
                    else:
                        high_price.append(_row.high)

                pos_lst.append(pos)

            position['pos' + str(i)] = pos_lst
            position = position.reset_index(drop=True)[['trade_date', 'pos' + str(i)]]
            pos_df = pos_df.merge(position, on=['trade_date'], how='outer')
        pos_df = pos_df.fillna(value=0)
        pos_df = pos_df.sort_values(['trade_date']).set_index(['trade_date'])
        pos_df['position'] = pos_df.sum(axis=1) / len(para_lst)
        pos_df = pos_df.reset_index(drop=False).merge(index_hq, on=['trade_date']).sort_values(['trade_date']) \
            .assign(close_1=lambda df: df.close.shift(1)).dropna()
        # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        # ax1 = ax.twinx()
        pos_df.index = pd.to_datetime(pos_df['trade_date'])

        # pos_df[['close']].plot(ax=ax1, figsize=(9, 6), kind='line', style=['k-'])
        # pos_df[['position']].plot(kind='area', grid=True, ax=ax, figsize=(9, 7), rot=60, style=['y'])
        #
        # title_str = '%s' % (index_name)
        # plt.savefig(fold + 'fig/' + title_str[:6] + '_position.png')
        # plt.show()
        position = pos_df
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
            index_name[:6], sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)

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

    sharpe_ratio = yearsharpRatio(chg_df['net'].tolist(), 1)
    sharpe = yearsharpRatio(chg_df['close_net'].tolist(), 1)
    ann_return = annROR(chg_df['net'].tolist(), 1)
    max_drawdown = maxRetrace(chg_df['net'].tolist(), 1)
    title_str = 'profolio sharpe %.2f idx_sharp %.2f ann_return %.2f max_drawdown %.2f' % (
        sharpe_ratio, sharpe, 100 * ann_return, 100 * max_drawdown)
    chg_df.ix[:, ['net', 'close_net']].plot()
    plt.title(title_str)
    plt.savefig(fold + 'fig/' + 'profolio.png')
    plt.show()
