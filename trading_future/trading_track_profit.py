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
import tkinter
import tkinter.messagebox


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
    long_margin = 50000
    short_margin = 50000
    level = 10
    long_value = long_margin * level
    short_value = short_margin * level
    long_code_lst = ['SC2009']
    short_code_lst = ['PP2009']
    # long_cost_lst = [311.16]
    long_cost_lst = [306.02]
    short_cost_lst = [6962.27]
    long_volume = [1]
    short_volume = [-11]
    long_contract = [Future(symbol=i[:-4]).contract_size for i in long_code_lst]
    short_contract = [Future(symbol=i[:-4]).contract_size for i in short_code_lst]
    lst = []
    long_value_ini = np.sum([long_cost_lst[i] * long_contract[i] * long_volume[i] for i in range(len(long_code_lst))])
    short_value_ini = np.sum([short_cost_lst[i] * short_contract[i] * short_volume[i] for i in range(len(short_code_lst))])
    for x in range(1000000000):
        # print(x)
        lst = []
        long_value_now = 0
        short_value_now = 0

        for i in range(len(long_code_lst)):
            index_code = long_code_lst[i]
            contract = long_contract[i]
            data = Future_ex(symbol=index_code, rd=rd).pub()
            if data is None:
                continue
            else:
                price_now = data['LastPrice']
                print(price_now)
                long_value_now = long_value_now + price_now * long_contract[i] * long_volume[i]

        for i in range(len(short_code_lst)):
            index_code = short_code_lst[i]
            contract = short_contract[i]
            data = Future_ex(symbol=index_code, rd=rd).pub()
            if data is None:
                continue
            else:
                price_now = data['LastPrice']
                print(price_now)
                short_value_now = short_value_now + price_now * short_contract[i] * short_volume[i]
        long_chng = long_value_now/long_value_ini - 1
        short_chng = (short_value_now-short_value_ini)/np.abs(short_value_ini)

        df = pd.DataFrame([[long_chng, short_chng, long_chng + short_chng]], columns=['long', 'short', 'net_profit'])
        print(datetime.datetime.now())
        print(df)
        if df.net_profit.tolist()[0] < -0.15:
            get_alert_info(df, '净亏损超6%：')


