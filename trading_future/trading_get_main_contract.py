#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 16:37
# @Author  : jwliu
# @Site    :
# @Software: PyCharm
import datetime
import pandas as pd
import numpy
import time
import data_engine.global_variable as global_variable
from data_engine.instrument.instrument import Instrument
from data_engine.data_factory import DataFactory
import redis
import json
import threading
from data_engine.instrument.future import Future
from data_engine.instrument.product import Product
import data_engine.setting as setting
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE



if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = myclient
    # data = jzmongo['MARKET']['continuous_contract'].find(filter={"by": 'open_init'})
    # df = pd.DataFrame(data)

    # opt_weight = jzmongo['portfolio']['strategy_opt_weight'].find(filter={'strategy': 'YMJH-Daily'})
    # df = pd.DataFrame(opt_weight)
    # lst = []
    # for method, group in df.groupby(['aggtoken', 'date']):
    #     row = []
    #     row.extend(method)
    #     row.append(group.weight.mean())
    #     lst.append(row)
    # ret = pd.DataFrame(lst, columns=['aggtoken', 'date', 'weight'])
    # print(ret)
    # df.to_csv('e:/df.csv')



    fold = 'e:/kdj_macd/'
    fold_data = 'e:/kdj_macd/data/'
    long_margin = 50000
    short_margin = 50000
    level = 10
    long_value = long_margin * level
    short_value = short_margin * level

    start_day = '2020-03-27'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    long_code_lst = ['SC2005']
    short_code_lst = ['L2009', 'PP2009']
    symbol_lst = DataFactory().get_product_info().index.tolist()
    print(pd.DataFrame(symbol_lst))
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                   'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    symbol_lst = ['ZN', 'CU', 'RU', 'SC', 'PP', 'HC', 'FG']
    lst = {}
    for symbol in symbol_lst:
        p = Product(symbol)
        p.list_futures()
        p.load_hq()

        try:
            # print(p.max_open_int_symbol())
            lst[symbol] = [p.max_open_int_symbol(), p.max_volume_symbol()]
        except:
            continue
    df = pd.DataFrame.from_dict(lst, orient='index', columns=['contract_open', 'contract_volume'])
    print(df)
    df.to_csv('G:/trading/main_contract.csv')
