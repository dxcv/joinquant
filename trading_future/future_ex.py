#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 16:37
# @Author  : jwliu
# @Site    : 
# @Software: PyCharm
import datetime
import pandas
import numpy
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
                        self.call_back(data=json.loads(data[2]))
                    else:
                        print(data[0])
                except Exception as e:
                    pass

    def call_back(self,data):
        # print(data)
        print(data['TRADE_CODE'],data['LastPrice'],data['AskPrice1'],data['BidPrice1'],data['TimeIndex'])


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', MONGDB_IP='192.168.2.201', MONGDB_USER='juzheng',
                       DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
                       , logging_level=global_variable.logging.INFO)
    rd = redis.Redis('192.168.1.36')

    symbol_list = ['SC2005','L2009','PP2009','BU2009','TA2009']
    f_list = [Future_ex(symbol=x,rd=rd) for x in symbol_list]
    for each in f_list:
        each.start()
    for each in f_list:
        each.join()
