#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 16:37
# @Author  : jwliu
# @Site    :
# @Software: PyCharm
import datetime
import pandas as pd
import numpy as np
import time
import data_engine.global_variable as global_variable
from data_engine.instrument.instrument import Instrument
from data_engine.data_factory import DataFactory
import redis
import json
import threading
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import copy
import talib as tb
import tkinter
import tkinter.messagebox

# auth('18610039264', 'zg19491001')
style.use('ggplot')
from tqsdk import TqApi, TqSim, TqAccount
auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)
from data_engine.data_factory import DataFactory
from data_engine.instrument.future import Future
import data_engine.setting as setting
from trading_simulate.trading_fuction import Trading
from email_fuction import send_email


def get_normal_future_index_code(code_lst):
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index

    temp = temp[['index_code', 'name']]
    code_dic = {}
    code_df = pd.DataFrame(code_lst, columns=['name'])
    code_df = code_df.merge(temp, on=['name']).set_index(['name'])

    for idx, _row in code_df.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic, code_lst


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    if sec in ['CU8888.XSGE', 'AU8888.XSGE', 'ZN8888.XSGE', 'SC8888.XINE']:
        if sec == 'CU8888.XSGE':
            symble = 'HG'
        elif sec == 'ZN8888.XSGE':
            symble = 'ZSD'
        elif sec == 'SC8888.XINE':
            symble = 'OIL'
        else:
            symble = 'GC'
        temp = finance.run_query(query(finance.FUT_GLOBAL_DAILY).filter(
            finance.FUT_GLOBAL_DAILY.code == symble, finance.FUT_GLOBAL_DAILY.day >= sday,
            finance.FUT_GLOBAL_DAILY.day <= eday)) \
            .rename(columns={'day': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))[
            ['open', 'high', 'low', 'close', 'date_time', 'volume']]

    else:
        temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                         count=None).reset_index() \
            .rename(columns={'index': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

    temp['stock_code'] = sec

    return temp


def get_alert_info(df, txt):
    if len(df) > 0:
        info_txt = txt
        for idx, row in df.iterrows():
            info_txt = info_txt + '净盈亏: ' + str(np.around(row.net_profit, 4)) + '; 多头盈亏: ' +\
                       str(np.around(row.long, 4)) + ' 空头盈亏: ' + \
                       str(np.around(row.short, 4)) + ';'
        print(info_txt)
        tkinter.messagebox.showinfo('提示', info_txt)


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour > 14:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


if __name__ == '__main__':
    from alarm_module import LogAsyncService
    from alarm_module import logger_async
    import logging

    receiver = ['xiahutao@163.com', '3467518502@qq.com', '897174480@qq.com']
    # with LogAsyncService() as log:
    #     try:
    #         # send_email(df, '完成1次', receiver)
    #         logger_async.log(__name__, logger_async.critical, '完成第一次交易')
    #     except Exception as e:
    #         print(str(e))

    DataFactory.config(MONGDB_PW='jz2018*', MONGDB_IP='192.168.2.201', MONGDB_USER='juzheng',
                       DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       , logging_level=global_variable.logging.INFO)
    rd = redis.Redis('192.168.1.36')

    # api = TqApi(TqAccount("G国泰君安", "85030120", "jz04282020"), web_gui=True)
    # Trd = Trading(api)
    hold_code_lst = ['sc2007', 'sc2012']
    start_day = datetime.date.today().strftime('%Y-%m-%d')
    # start_day = '2020-05-18'
    today = datetime.date.today()
    calen = get_trade_days(count=5)
    calen = list(calen)
    calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, today)
    end_day = EndDate
    print(end_day)

    normalize_code_future, index_code_lst = get_normal_future_index_code(hold_code_lst)
    for index_code in index_code_lst:
        symble = normalize_code_future[index_code]

    long_code_lst = ['sc2007']
    short_code_lst = ['sc2012']
    # long_cost_lst = [311.16]
    long_cost_lst = [267.4]
    short_cost_lst = [315]
    long_volume = [1]
    short_volume = [-1]
    long_contract = [Future(symbol=i[:-4].upper()).contract_size for i in long_code_lst]
    short_contract = [Future(symbol=i[:-4].upper()).contract_size for i in short_code_lst]
    lst = []
    long_value_ini = np.sum([long_cost_lst[i] * long_contract[i] * long_volume[i] for i in range(len(long_code_lst))])
    short_value_ini = np.sum([short_cost_lst[i] * short_contract[i] * short_volume[i] for i in range(len(short_code_lst))])
    times1 = 0
    times2 = 0
    times3 = 0
    while datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') < '2020-05-21 02:30:00':

        lst = []
        long_value_now = 0
        short_value_now = 0

        for i in range(len(long_code_lst)):
            index_code = long_code_lst[i]
            contract = long_contract[i]
            data = stock_price_cgo(normalize_code_future[index_code], start_day, end_day)[
                ['date_time', 'close', 'high', 'low', 'open']]
            print(data)
            price_now = data.close.tolist()[-1]
            long_value_now = long_value_now + price_now * long_contract[i] * long_volume[i]
        for i in range(len(short_code_lst)):
            index_code = short_code_lst[i]
            contract = short_contract[i]
            data = stock_price_cgo(normalize_code_future[index_code], start_day, end_day)[
                ['date_time', 'close', 'high', 'low', 'open']]
            print(data)
            price_now = data.close.tolist()[-1]

            short_value_now = short_value_now + price_now * short_contract[i] * short_volume[i]
        long_chng = long_value_now/long_value_ini - 1
        short_chng = (short_value_now-short_value_ini)/np.abs(short_value_ini)

        df = pd.DataFrame([[long_chng, short_chng, long_chng + short_chng]], columns=['long', 'short', 'net_profit'])
        print(datetime.datetime.now())
        print(df)

        print(df.net_profit.tolist()[0])

        # if (df.net_profit.tolist()[0] < -0.01) and times1 == 0:
        #     try:
        #         send_email(df, '完成1次', receiver)
        #         # logger_async.log(__name__, logger_async.critical, '完成第一次交易')
        #     except Exception as e:
        #         print(str(e))
        #     try:
        #         api = TqApi(TqAccount("G国泰君安", "85030120", "jz04282020"), web_gui=True)
        #         Trd = Trading(api)
        #         order = Trd.insert_order_sk_limit('INE.sc2012', 1)
        #         order = Trd.insert_order_bk_limit('INE.sc2007', 1)
        #         times1 += 1
        #         break
        #     except Exception as e:
        #         print(str(e))

            # get_alert_info(df, '净亏损超4%：')
        # if (df.net_profit.tolist()[0] < -0.02) and times2 == 0:
        #     order = Trd.insert_order_sk_limit('INE.sc2012', 1)
        #     order = Trd.insert_order_bk_limit('INE.sc2007', 1)
        #     times2 += 1
        #     try:
        #         send_email(df, '完成2次', receiver)
        #         # logger_async.log(__name__, logger_async.critical, '完成第一次交易')
        #     except Exception as e:
        #         print(str(e))
        # if (df.net_profit.tolist()[0] < -0.03) and times3 == 0:
        #     order = Trd.insert_order_sk_limit('INE.sc2012', 1)
        #     order = Trd.insert_order_bk_limit('INE.sc2007', 1)
        #     times3 += 1
        #     try:
        #         send_email(df, '完成3次', receiver)
        #         # logger_async.log(__name__, logger_async.critical, '完成第一次交易')
        #     except Exception as e:
        #         print(str(e))
        time.sleep(5)



