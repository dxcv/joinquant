# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import sys
print(sys.path)
sys.path.append('C:\\Users\\Administrator\\PycharmProjects\\joinquant')  # 新加入的
print(sys.path)
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


def get_tcs_pos_df_all(class_lst, symbols_dict, index_code_lst, ma_period_dict, k_period_dict,
                       index_hq_dic, hq_last_date, volLookback, targetVol):
    pos_df_all = pd.DataFrame(columns=['trade_date'])
    ma_period_dic = {}
    k_period_dic = {}
    for clas in class_lst:
        symbols = symbols_dict[clas]
        for symbol in symbols:
            ma_period_dic[symbol] = ma_period_dict[clas]
            k_period_dic[symbol] = k_period_dict[clas]
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        pos_df = pd.DataFrame(columns=['trade_date'])

        index_hq = index_hq_dic[index_code]
        index_hq = index_hq[index_hq['trade_date'] <= hq_last_date]
        ma_period_lst = ma_period_dic[index_code]
        k_lst = k_period_dic[index_code]
        for i in range(len(ma_period_lst)):
            ma_period = ma_period_lst[i]
            k = k_lst[i]
            position = index_hq \
                .assign(ma=lambda df: tb.MA(df['close'].values, ma_period)) \
                .assign(delta=lambda df: abs(df.close - df.close.shift(1))) \
                .assign(delta_ma=lambda df: tb.MA(df['delta'].values, ma_period)) \
                .assign(up=lambda df: df.ma + k * df.delta_ma) \
                .assign(down=lambda df: df.ma - k * df.delta_ma) \
                .assign(price_return=lambda df: df.close / df.close.shift(1) - 1)
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
                condition_l = _row.close > _row.up
                condition_s = _row.close < _row.down
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
        pos_df[index_code] = pos_df.sum(axis=1) / len(k_lst)
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
    aum = 16000000
    balance = 5
    strategy_id = 'tcs'
    fold_path = 'G://trading_strategy//'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    receiver = ['xiahutao@163.com', '3467518502@qq.com']
    today = datetime.date.today()
    class_lst = ['Grains', 'Chem', 'BaseMetal', 'Bulks', 'Equity', 'Bonds', 'PreciousMetal', 'LightIndustry',
                 'Oil', 'Grease', 'SoftComm', 'Food']
    index_code_lst = ['SC', 'J', 'TA', 'I', 'MA', 'RU', 'ZC', 'P', 'SF', 'CF', 'NI', 'TF', 'IF', 'SM', 'PB', 'T', 'SR',
                   'AL', 'BU', 'AU', 'AP', 'FU', 'C']  # sharp>0.2所有品种
    # index_code_lst = ['I']
    normalize_code_future = get_normal_future_index_code()
    symbols_dict = {'Grains': ['C', 'A', 'B'],  # 农产品
                    'Grease': ['M', 'RM', 'Y', 'P', 'OI'],  # 油脂油料
                    'Chem': ['L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA'],  # 化工
                    'Oil': ['SC', 'FU'],
                    'BaseMetal': ['AL', 'ZN', 'CU', 'PB', 'NI', 'SN'],  # 金属
                    'Bulks': ['J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM'],  # 黑色系
                    'LightIndustry': ['SP', 'FG'],  # 轻工业
                    'Equity': ['IF', 'IH', 'IC'],  # 股指
                    'Bonds': ['T', 'TF'],  # 债券
                    'PreciousMetal': ['AG', 'AU'],  # 贵金属
                    'SoftComm': ['CF', 'CS', 'SR'],  # 软商品
                    'Food': ['JD', 'AP', 'CJ'],  # 农副产品
                    }  # 所有品种
    ma_period_dict = {'Grains': [10, 10, 15, 15, 20],
                      'Chem': [30, 70, 45, 25, 15],
                      'BaseMetal': [10, 15, 35, 15, 30],
                      'Bulks': [35, 20, 95, 15, 35],
                      'Equity': [10, 15, 45, 95, 80],
                      'Bonds': [20, 35, 80, 30],
                      'PreciousMetal': [25, 20, 10, 10, 20],
                      'LightIndustry': [10, 25, 95, 55],
                      'Oil': [10, 75, 20, 40, 10],
                      'Grease': [10, 95, 10, 15, 65],
                      'SoftComm': [15, 55, 20, 25, 95],
                      'Food': [10, 85, 70, 30]}
    k_dict = {'Grains': [0.7, 0.7, 2.1, 4.7, 1.1],
              'Chem': [3.9, 0.7, 1.7, 3.5, 0.7],
              'BaseMetal': [1.3, 2.1, 2.5, 4.7, 2.3],
              'Bulks': [1.3, 1.5, 4.7, 2.3, 3.9],
              'Equity': [1.7, 1.7, 2.5, 4.7, 4.7],
              'Bonds': [3.7, 4.1, 4.7, 3.7],
              'PreciousMetal': [0.9, 0.9, 0.7, 0.9, 2.5],
              'LightIndustry': [1.9, 4.7, 4.7, 1.9],
              'Oil': [0.7, 4.7, 2.7, 1.7, 0.7],
              'Grease': [0.7, 4.3, 2.3, 4.5, 3.3],
              'SoftComm': [1.5, 2.1, 4.5, 1.3, 4.7],
              'Food': [1.1, 0.7, 3.1, 0.9]}
    # N = 130
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
        pos_df_all_ymjh = get_tcs_pos_df_all(class_lst, symbols_dict, index_code_lst, ma_period_dict, k_dict,
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
        trading_info.to_csv('G://trading_strategy//' + 'position_tcs_' + hq_last_date + '.csv')
        subject = date + strategy_id

        send_email(trading_info, subject, receiver)
