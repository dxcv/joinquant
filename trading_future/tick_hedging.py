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
        print(data['TRADE_CODE'],data['LastPrice'],data['AskPrice1'],data['BidPrice1'],data['TimeIndex'])
        return data


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
    long_code_lst = ['SC2005']
    short_code_lst = ['L2009', 'PP2009', 'TA2009', 'BU2009']
    long_cost_lst = [251]
    short_cost_lst = [5765, 5999]
    long_contract = [Future(symbol=i[:-4]).contract_size for i in long_code_lst]
    short_contract = [Future(symbol=i[:-4]).contract_size for i in short_code_lst]
    for x in range(1000000000):

        lst = []
        for i in range(len(long_code_lst)):
            index_code = long_code_lst[i]
            contract = long_contract[i]
            data = Future_ex(symbol=index_code, rd=rd).pub()
            if data is None:
                continue
            else:
                price_now = data['LastPrice']
                long_volume = long_value / len(long_code_lst) / price_now / contract
                row = []
                row.append(index_code)
                row.append(long_volume)
                row.append(long_volume * len(long_code_lst))
                row.append(price_now)
                row.append(contract)
                lst.append(row)
        for i in range(len(short_code_lst)):
            index_code = short_code_lst[i]
            contract = short_contract[i]
            data = Future_ex(symbol=index_code, rd=rd).pub()
            if data is None:
                continue
            else:
                price_now = data['LastPrice']
                long_volume = - short_value / len(short_code_lst) / price_now / contract
                row = []
                row.append(index_code)
                row.append(long_volume)
                row.append(long_volume * len(short_code_lst))
                row.append(price_now)
                row.append(contract)
                lst.append(row)
        df = pd.DataFrame(lst, columns=['symbol', 'volume', 'volume_all', 'price', 'contract'])
        df['volume_adj'] = np.around(df['volume'])
        df['volume_all'] = df['volume_all'] / df['volume_all'].tolist()[0]
        # df['volume_all'] = np.around(df['volume_all'])
        df['every_val'] = df['contract'] * df['price']
        df['value_adj'] = df['volume_adj'] * df['contract'] * df['price']
        print(datetime.datetime.now())
        print(df[['symbol', 'volume', 'volume_adj', 'volume_all', 'value_adj', 'every_val']]\
              .set_index(['symbol']))
        print('多头头寸：', df[df['value_adj'] > 0].value_adj.sum())
        print('空头头寸：', df[df['value_adj'] < 0].value_adj.sum())
        print('净头寸：', df.value_adj.sum())


