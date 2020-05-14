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
from data_engine.instrument.future import Future
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

auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)
from data_engine.data_factory import DataFactory
from data_engine.instrument.future import Future
import data_engine.setting as setting

class Future_ex(Future):
    def __init__(self,symbol,by_ctp_instrument=False,info=None,rd=None):
        Future.__init__(self,symbol=symbol,by_ctp_instrument=by_ctp_instrument,info=info)
        self.rd = rd
        self.run = True
        self.t = None
        print('init ' + self.ctp_symbol)

    def start(self):
        self.run = True
        t = threading.Thread(target=self.pub)
        t.setDaemon(True)
        t.start()
        self.t = t
        return self

    def end(self):
        self.run = False
        return self

    def join(self):
        self.t.join()

    @property
    def chan_sub(self):
        return 'P_' + self.ctp_symbol

    def pub(self):
        if isinstance(self.rd,redis.Redis):
            pub = self.rd.pubsub()
            pub.subscribe(self.chan_sub)
            while self.run:
                data = pub.parse_response(block=False, timeout=60)
                if data is None:
                    time.sleep(0.5)
                    continue
                try:
                    if data[0] == b'message':
                        data = self.call_back(data=json.loads(data[2]))
                        return data
                    else:
                        print(data[0])
                except Exception as e:
                    pass

    def call_back(self,data):
        # print(data)
        # print(data['TRADE_CODE'],data['LastPrice'],data['AskPrice1'],data['BidPrice1'],data['TimeIndex'])
        return data


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


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', MONGDB_IP='192.168.2.201', MONGDB_USER='juzheng',
                       DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       , logging_level=global_variable.logging.INFO)
    rd = redis.Redis('192.168.1.36')
    hold_code_lst = ['sc2009', 'pp2009']
    start_day = datetime.date.today().strftime('%Y-%m-%d')
    end_day = datetime.date.today().strftime('%Y-%m-%d')

    normalize_code_future, index_code_lst = get_normal_future_index_code(hold_code_lst)
    for index_code in index_code_lst:
        symble = normalize_code_future[index_code]

    long_code_lst = ['sc2009']
    short_code_lst = ['pp2009']
    # long_cost_lst = [311.16]
    long_cost_lst = [306.02]
    short_cost_lst = [6962.27]
    long_volume = [1]
    short_volume = [-1]
    long_contract = [Future(symbol=i[:-4].upper()).contract_size for i in long_code_lst]
    short_contract = [Future(symbol=i[:-4].upper()).contract_size for i in short_code_lst]
    lst = []
    long_value_ini = np.sum([long_cost_lst[i] * long_contract[i] * long_volume[i] for i in range(len(long_code_lst))])
    short_value_ini = np.sum([short_cost_lst[i] * short_contract[i] * short_volume[i] for i in range(len(short_code_lst))])

    for x in range(1000000000):
        print(x)
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
        if df.net_profit.tolist()[0] < -0.08:
            get_alert_info(df, '净亏损超4%：')
        time.sleep(5)



