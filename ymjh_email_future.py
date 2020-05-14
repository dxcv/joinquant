# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
from jqdatasdk import *
import copy
from email_fuction import send_email
from trading_future.future_singleton import Future
# auth('18610039264', 'zg19491001')
auth('15658001226', 'taiyi123')
import datetime
import talib as tb
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='post',
                     count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


def stock_code_lst(sec):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['wind_index'].read(sec + '_component')
    temp = temp[temp['date'] == temp['date'].max()]
    return temp.code.tolist()


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour >= 15:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    listDATE = calen[-125]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, listDATE, StartDate, str(hq_last_date)[:10]


def cap_vol_by_rolling(vol, target_vol):
    idxs = vol.index
    for idx in range(len(idxs)):
        curDate = idxs[idx]
        vol[curDate] = max(vol[curDate], target_vol)
    return vol


def get_ymjh_pos_df_all(class_lst, symbols_dict, index_code_lst, s_period_dict, l_period_dict,
                        index_hq_dic, hq_last_date, volLookback, targetVol):
    pos_df_all = pd.DataFrame(columns=['trade_date'])
    s_period_dic = {}
    l_period_dic = {}
    for clas in class_lst:
        symbols = symbols_dict[clas]
        for symbol in symbols:
            s_period_dic[symbol] = s_period_dict[clas]
            l_period_dic[symbol] = l_period_dict[clas]
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        pos_df = pd.DataFrame(columns=['trade_date'])

        index_hq = index_hq_dic[index_code]
        index_hq = index_hq[index_hq['trade_date'] <= hq_last_date]
        N1_lst = s_period_dic[index_code]
        N2_lst = l_period_dic[index_code]
        for i in range(len(N1_lst)):
            N1 = N1_lst[i]
            N2 = N2_lst[i]
            position = index_hq \
                .assign(HH_s=lambda df: tb.MAX(df.high.values, N1)) \
                .assign(LL_s=lambda df: tb.MIN(df.low.values, N1)) \
                .assign(HH_l=lambda df: tb.MAX(df.high.values, N2)) \
                .assign(LL_l=lambda df: tb.MIN(df.low.values, N2)) \
                .assign(ma_s=lambda df: (df.HH_s + df.LL_s) / 2) \
                .assign(ma_l=lambda df: (df.HH_l + df.LL_l) / 2) \
                .assign(ma_s1=lambda df: df.ma_s.shift(1)) \
                .assign(ma_l1=lambda df: df.ma_l.shift(1)) \
                .assign(ave_p=lambda df: (2 * df.close + df.high + df.low) / 4) \
                .assign(price_return=lambda df: df.close/df.close.shift(1) - 1)
            if volLookback != 0:
                realizedVol = position['price_return'].ewm(
                    span=volLookback, ignore_na=True, adjust=False).std(bias=True) * (252 ** 0.5)
                # if index_code not in ['T', 'TF']:
                realizedVol = cap_vol_by_rolling(realizedVol, targetVol)
                riskScaler = targetVol / realizedVol
                position['riskScaler'] = riskScaler
            else:
                position['riskScaler'] = 1
            _signal = 0
            pos_lst = []
            for idx, _row in position.iterrows():
                condition_l = ((_row.ma_s1 == _row.ma_l1) and (_row.ma_s > _row.ma_l) and (
                        _row.ave_p >= _row.ma_s)) or ((_row.ma_s1 < _row.ma_l1) and (
                        _row.ma_s > _row.ma_l) and (_row.ave_p >= min(_row.ma_s, _row.ma_l)))
                condition_s = (_row.ma_s1 > _row.ma_l1) and (_row.ma_s < _row.ma_l) and (
                        _row.ave_p <= max(_row.ma_s, _row.ma_l))
                if _signal == 0:
                    if condition_l:
                        _signal = _row.riskScaler
                    elif condition_s:
                        _signal = -_row.riskScaler
                elif _signal > 0:
                    if condition_s:
                        _signal = -_row.riskScaler
                elif _signal < 0:
                    if condition_l:
                        _signal = _row.riskScaler
                pos_lst.append(_signal)
            position['pos' + str(i)] = pos_lst
            position = position.reset_index(drop=True)[['trade_date', 'pos' + str(i)]]
            pos_df = pos_df.merge(position, on=['trade_date'], how='outer')
        pos_df = pos_df.dropna().sort_values(['trade_date']).set_index(['trade_date'])
        pos_df[index_code] = pos_df.sum(axis=1) / len(N1_lst)
        pos_df_all = pos_df_all.merge(
            pos_df.reset_index(drop=False)[['trade_date', index_code]], on=['trade_date'], how='outer')
    return pos_df_all


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=True, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'}) \
        .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    temp['stock_code'] = sec
    return temp


def get_signal(signal, aum, balance, EndDate, close_dict):

    signal['symbol'] = signal.index
    symbol_lst = signal.symbol.tolist()
    porfolio = Future()
    main_contract_dict = porfolio.get_main_symbol(product=symbol_lst, date=EndDate)
    main_contract = pd.DataFrame(main_contract_dict).T[['main_contract']]
    contract_lst = main_contract.main_contract.tolist()
    ExchangeID_dict = porfolio.get_ExchangeID(contract_lst=contract_lst)
    ExchangeInstID_dict = porfolio.get_ExchangeInstID(contract_lst=contract_lst)
    VolumeMultiple_dict = porfolio.get_VolumeMultiple(contract_lst)

    main_contract['symbol'] = main_contract.index
    signal_dict = {}
    for symbol in symbol_lst:
        main_contract = main_contract_dict[symbol]['main_contract']
        trading_code = ExchangeID_dict[main_contract]['ExchangeID'] + '.' + ExchangeInstID_dict[main_contract][
            'ExchangeInstID']
        signal_dict[symbol] = {
            'symbol': symbol, 'trading_code': trading_code, 'weight': signal.loc[symbol]['weight'],
            'last_price': close_dict[symbol],
            'VolumeMultiple': VolumeMultiple_dict[main_contract]['VolumeMultiple']
        }
    trading_info = pd.DataFrame(signal_dict).T
    trading_info['position'] = aum * balance / len(symbol_lst) * trading_info['weight'] / trading_info[
        'last_price'] / trading_info['VolumeMultiple']
    # trading_info['position'] = trading_info['position'].apply(lambda x: int(np.around(x, 0)))
    print(trading_info)
    return trading_info


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    aum = 1000000
    balance = 5
    strategy_id = 'ymjh'
    fold_path = 'G://trading_strategy//'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    receiver = ['xiahutao@163.com', '3467518502@qq.com']
    today = datetime.date.today()
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'SoftComm', 'Food']
    index_code_lst = ['AP', 'HC', 'J', 'TA', 'SC', 'I', 'RU', 'TF', 'RB', 'ZC', 'M', 'MA', 'AU', 'AG', 'NI', 'C', 'CU',
                   'V', 'BU', 'SF', 'PB', 'A', 'T']  # sharp>0.2所有品种
    # index_code_lst = ['I']
    normalize_code_future = get_normal_future_index_code()
    symbols_dict = {'Grains': ['C', 'A', 'M', 'Y', 'P', 'OI', 'B', 'RM', 'C'],
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'FG', 'SF', 'SM'],
                    'Equity': ['IF', 'IH', 'IC'],
                    'Bonds': ['T', 'TF'],
                    'PreciousMetal': ['AG', 'AU'],
                    'SoftComm': ['CF', 'CS', 'SR'],
                    'Food': ['JD', 'AP', 'CJ'],  # 农副产品
                    }
    s_period_dict = {'Grains': [28, 8, 13, 25, 8],
                     'Chem': [5, 5, 10, 6, 5],
                     'BaseMetal': [16, 6, 7, 9, 5],
                     'Bulks': [7, 6, 6, 7, 28],
                     'Equity': [16, 26, 5, 24, 19],
                     'Bonds': [7, 13, 4, 22],
                     'PreciousMetal': [6, 26, 9, 18, 26],
                     'SoftComm': [9, 5, 18, 15, 4],
                     'Food': [4, 7, 11, 8]}
    l_period_dict = {'Grains': [45, 18, 27, 30, 23],
                     'Chem': [32, 46, 44, 33, 20],
                     'BaseMetal': [44, 17, 29, 14, 41],
                     'Bulks': [35, 38, 29, 32, 62],
                     'Equity': [65, 50, 50, 32, 26],
                     'Bonds': [20, 62, 17, 29],
                     'PreciousMetal': [23, 41, 14, 41, 56],
                     'SoftComm': [65, 56, 65, 20, 62],
                     'Food': [14, 20, 20, 47]}
    N = 100
    num = 0
    bars = 252
    volLookback = 20
    targetVol = 0.1
    calen = get_trade_days(count=bars)
    calen = list(calen)
    if today in calen:
        calen, next_tradeday, EndDate, listDATE, StartDate, hq_last_date = get_date(calen, today)
        index_hq_dic = {}
        EndDate = EndDate.strftime('%Y-%m-%d')
        date = EndDate
        close_dict = {}
        for index_code in index_code_lst:
            code = normalize_code_future[index_code]
            index_hq = stock_price(code, 'daily', StartDate, hq_last_date)
            index_hq_dic[index_code] = index_hq
            close_dict[index_code] = index_hq[index_hq['trade_date'] == hq_last_date].close.tolist()[0]
        pos_df_all_ymjh = get_ymjh_pos_df_all(class_lst, symbols_dict, index_code_lst, s_period_dict, l_period_dict,
                        index_hq_dic, hq_last_date, volLookback, targetVol)
        print(pos_df_all_ymjh)

        res = pos_df_all_ymjh.sort_values(by='trade_date', ascending=False)
        res.index = range(len(res))
        res_n = res.copy()
        res_n = res_n[res_n['trade_date'] == hq_last_date].drop(['trade_date'], axis=1).T
        res_n.columns = ['weight']
        print(res_n)
        res_n.to_csv(fold_path + 'temp//' + strategy_id + '_' + hq_last_date + '.csv')
        # send_email(res_n, date, receiver)

        trading_info = get_signal(res_n, aum, balance, EndDate, close_dict)
        trading_info.to_csv('G://trading_strategy//' + 'position_ymjh_' + hq_last_date + '.csv')
        subject = date + strategy_id
        send_email(trading_info, subject, receiver)




