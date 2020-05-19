# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 13:26
# @Author  : zhangfang
from tqsdk import TqApi, TqSim, TqAccount
import pandas as pd
import numpy as np
import time
from jqdatasdk import *
import datetime
from trading_future.future_singleton import Future
from trading_simulate.trading_fuction import Trading

# auth('18610039264', 'zg19491001')
auth('15658001226', 'taiyi123')


if __name__ == '__main__':
    signal_path = 'G://trading_strategy//'
    aum = 1000000
    balance = 10
    bars = 5
    # api = TqApi(TqAccount("simnow", "163919", "yo193846"), web_gui=True)
    # api = TqApi(TqAccount("快期模拟", "519518384@qq.com", "zf1991"), web_gui=True)
    api = TqApi(TqAccount("G国泰君安", "85030120", "jz04282020"), web_gui=True)
    Trd = Trading(api)
    account = api.get_account()
    quote = api.get_quote("INE.sc2012")  # 获取SHFE.cu1812合约的10秒K线
    time_allowed_lst = quote.trading_time.day
    time_allowed_lst.extend(quote.trading_time.night)
    print(time_allowed_lst)

    times = 0

    # while datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') > '2020-05-18 09:00:00' and times == 0:
    #     print(times)

    for i in range(3):
        order = Trd.insert_order_sk_limit('INE.sc2012', 1)
        order = Trd.insert_order_bk_limit('INE.sc2007', 1)
        time.sleep(10)

    times += 1

    # while api.wait_update():
    #     print(account.float_profit)
    #
    #     time.sleep(10)