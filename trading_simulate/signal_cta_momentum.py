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


def vol_estimator_garch(data_df, st=25, lt=252 * 3):  # 250*5):
    st_span = st  # min(st,len(data_df))
    lt_span = lt  # min(lt,len(data_df))
    # print(st_span, lt_span,st)
    st_vol = data_df.ewm(span=st_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
    lt_vol = data_df.ewm(span=lt_span, ignore_na=True, min_periods=st, adjust=False).std(bias=True)
    decay_rate = 0.8
    vol = st_vol * decay_rate + lt_vol * (1 - decay_rate)
    # vol=self.cap_vol_by_rolling(vol)
    return vol


def get_resp_curve(x, method):
    resp_curve = pd.DataFrame()
    if method == 'gaussian':
        resp_curve = np.exp(-(x ** 2) / 4.0)
    return resp_curve


def get_momentum_pos_df_all(index_code_lst, SK, LK, index_hq_dic, hq_last_date, volLookback, targetVol):
    pos_df_all = pd.DataFrame(columns=['trade_date'])
    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        pos_df = pd.DataFrame(columns=['trade_date'])

        index_hq = index_hq_dic[index_code]
        index_hq = index_hq[index_hq['trade_date'] <= hq_last_date]
        data_daily = index_hq.assign(price_return=lambda df: df.close / df.close.shift(1) - 1)
        for k in range(len(SK)):

            volAdjRet = data_daily['price_return'] / data_daily['price_return'].ewm(span=SK[k], min_periods=SK[k],
                                                                                    adjust=False).std(bias=True)
            px_df = np.cumsum(volAdjRet)
            sig = px_df.ewm(span=SK[k], min_periods=SK[k]).mean() - px_df.ewm(span=LK[k], min_periods=SK[k]).mean()
            sig_normalized = sig / vol_estimator_garch(sig, 25)
            sig_resp = get_resp_curve(sig_normalized, 'gaussian')
            os_norm = 1.0 / 0.89
            sig = sig_normalized * sig_resp * os_norm
            realizedVol = data_daily['price_return'].ewm(span=volLookback, ignore_na=True, min_periods=20,
                                                         adjust=False).std(bias=True) * (252 ** 0.5)
            realizedVol = cap_vol_by_rolling(realizedVol, targetVol)
            riskScaler = targetVol / realizedVol
            scaledSig = riskScaler * sig
            data_daily['uk' + str(k)] = scaledSig
            data_daily['uk' + str(k)].fillna(method='pad', inplace=True)
        scaledSig = 1 / 5 * (
                    data_daily['uk0'] + data_daily['uk1'] + data_daily['uk2'] + data_daily['uk3'] + data_daily[
                'uk4'])
        leverageRaw = abs(scaledSig)
        leverageCapped = leverageRaw
        signOfSignal = scaledSig.apply(lambda x: np.sign(x))
        signalFinal = signOfSignal * leverageCapped
        data_daily['Scta'] = signalFinal
        pos_df = data_daily[['Scta', 'trade_date']].rename(columns={'Scta': index_code})
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
    strategy_id = 'momentum'
    fold_path = 'G://trading_strategy//'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    receiver = ['xiahutao@163.com', '3467518502@qq.com']
    today = datetime.date.today()

    index_code_lst = ['J', 'HC', 'RB', 'I', 'NI', 'TF', 'SM', 'AL', 'RU', 'MA', 'SR', 'P', 'TA', 'T', 'SC', 'IF', 'Y',
                      'FU', 'IH', 'AG', 'PB']  # sharp>0.2所有品种
    # index_code_lst = ['I']
    normalize_code_future = get_normal_future_index_code()

    s_period_dict = (4, 8, 16, 32, 64)
    l_period_dict = (8, 16, 32, 64, 128)
    N = 100
    num = 0
    bars = 252 * 4
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
        pos_df_all_momentum = get_momentum_pos_df_all(index_code_lst, s_period_dict, l_period_dict,
                        index_hq_dic, hq_last_date, volLookback, targetVol)
        print(pos_df_all_momentum)

        res = pos_df_all_momentum.sort_values(by='trade_date', ascending=False)
        res.index = range(len(res))
        res_n = res.copy()
        res_n = res_n[res_n['trade_date'] == hq_last_date].drop(['trade_date'], axis=1).T
        res_n.columns = ['weight']
        print(res_n)
        res_n.to_csv(fold_path + 'temp//' + strategy_id + '_' + hq_last_date + '.csv')
        # send_email(res_n, date, receiver)

        trading_info = get_signal(res_n, aum, balance, EndDate, close_dict)
        trading_info.to_csv('G://trading_strategy//' + 'position_momentum_' + hq_last_date + '.csv')
        subject = date + strategy_id
        send_email(trading_info, subject, receiver)




