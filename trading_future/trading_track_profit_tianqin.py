# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 10:23
# @Author  : zhangfang

from tqsdk import TqApi
from trading_future.future_singleton import Future
import pandas as pd
import datetime
import numpy as np
import time
import tkinter
import tkinter.messagebox
import copy


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
    # 创建api实例，设置web_gui=True生成图形化界面
    api = TqApi(web_gui=True)
    product_lst = ['SC', 'PP']

    hold_code_lst = ['sc2009', 'pp2009']
    info_lst = ['ExchangeID', 'VolumeMultiple']
    porfolio = Future()
    # tq_code_lst = [exchageID[i.upper()]['ExchangeID'] + '.' + i for i in hold_code_lst]
    # print(tq_code_lst)
    # 订阅 cu2002 合约的10秒线

    long_code_lst = ['sc2009']
    short_code_lst = ['pp2009']
    long_cost_lst = [306.02]
    short_cost_lst = [6962.27]
    long_volume = [1]
    short_volume = [-1]
    info_lst = ['ExchangeID', 'VolumeMultiple']
    long_product_lst = [i[:-4] for i in long_code_lst]
    short_product_lst = [j[:-4] for j in short_code_lst]
    symbol_lst = copy.deepcopy(long_product_lst)
    symbol_lst.extend(short_product_lst)
    contract_lst = copy.deepcopy(long_code_lst)
    contract_lst.extend(short_code_lst)
    exchageID = porfolio.get_contract_info(symbol_lst=symbol_lst, contract_lst=contract_lst, info_lst=info_lst)
    print(exchageID)
    long_contract = [exchageID[i.upper()]['VolumeMultiple'] for i in long_code_lst]
    short_contract = [exchageID[i.upper()]['VolumeMultiple'] for i in short_code_lst]
    print(long_contract, short_contract)
    lst = []
    long_value_ini = np.sum([long_cost_lst[i] * long_contract[i] * long_volume[i] for i in range(len(long_code_lst))])
    short_value_ini = np.sum(
        [short_cost_lst[i] * short_contract[i] * short_volume[i] for i in range(len(short_code_lst))])

    while True:

        lst = []
        long_value_now = 0
        short_value_now = 0

        for i in range(len(long_code_lst)):
            index_code = long_code_lst[i]
            contract = long_contract[i]
            price_now = api.get_quote(exchageID[index_code.upper()]['ExchangeID'] + '.' + index_code).last_price
            print(price_now)
            long_value_now = long_value_now + price_now * long_contract[i] * long_volume[i]

        for i in range(len(short_code_lst)):
            index_code = short_code_lst[i]
            contract = short_contract[i]
            price_now = api.get_quote(exchageID[index_code.upper()]['ExchangeID'] + '.' + index_code).last_price
            print(price_now)
            short_value_now = short_value_now + price_now * short_contract[i] * short_volume[i]
        long_chng = long_value_now / long_value_ini - 1
        short_chng = (short_value_now - short_value_ini) / np.abs(short_value_ini)

        df = pd.DataFrame([[long_chng, short_chng, long_chng + short_chng]], columns=['long', 'short', 'net_profit'])
        print(datetime.datetime.now())
        print(df)
        if df.net_profit.tolist()[0] < -0.12:
            get_alert_info(df, '净亏损超12%：')
        api.wait_update()

