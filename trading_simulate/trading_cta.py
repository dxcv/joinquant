# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 13:26
# @Author  : zhangfang
import sys

sys.path.append('C:\\Users\\Administrator\\PycharmProjects\\joinquant')  # 新加入的

from tqsdk import TqApi, TqSim, TqAccount
import pandas as pd
import numpy as np
import time
from jqdatasdk import *
import datetime
from trading_simulate.trading_fuction import Trading

# auth('18610039264', 'zg19491001')
auth('15658001226', 'taiyi123')


if __name__ == '__main__':
    signal_path = 'G://trading_strategy//'
    aum = 1000000
    balance = 10
    bars = 5
    # api = TqApi(TqAccount("快期模拟", "519518384@qq.com", "zf1991"), web_gui=True)
    api = TqApi(TqAccount("simnow", "163919", "yo193846"), web_gui=False)
    Trd = Trading(api)
    calen = get_trade_days(count=bars)
    today = datetime.date.today()
    calen = list(calen)
    if today in calen:
        calen, next_tradeday, EndDate, StartDate, hq_last_date = Trd.get_date(calen, today)

        temp_ymjh = pd.read_csv(signal_path + 'position_' + 'ymjh' + '_' + hq_last_date + '.csv', index_col=0)[
            ['VolumeMultiple', 'last_price', 'symbol', 'trading_code', 'position']]\
            .rename(columns={'position': 'position_ymjh'})
        temp_tcs = pd.read_csv(signal_path + 'position_' + 'tcs' + '_' + hq_last_date + '.csv', index_col=0)[
            ['VolumeMultiple', 'last_price', 'symbol', 'trading_code', 'position']]\
            .rename(columns={'position': 'position_tcs'})
        temp_momentum = pd.read_csv(signal_path + 'position_' + 'momentum' + '_' + hq_last_date + '.csv', index_col=0)[
            ['VolumeMultiple', 'last_price', 'symbol', 'trading_code', 'position']] \
            .rename(columns={'position': 'position_momentum'})
        trading_info = temp_tcs.merge(
            temp_ymjh, on=['VolumeMultiple', 'last_price', 'symbol', 'trading_code'], how='outer').merge(
            temp_momentum, on=['VolumeMultiple', 'last_price', 'symbol', 'trading_code'], how='outer').fillna(value=0)
        trading_info['position'] = trading_info['position_ymjh'] + trading_info['position_tcs'] + trading_info['position_momentum']
        print(trading_info)
        trading_info['position'] = trading_info['position'].apply(lambda x: int(np.around(x, 0)))
        trading_info.index = trading_info['trading_code']
        print(trading_info)
        code_lst = trading_info.trading_code.tolist()

        while True:
            print('==========================================================================================')
            orders = api.get_order()
            for oid, order in orders.items():
                if order.status == 'ALIVE':
                    print(order.status)
                    api.cancel_order(order)

            positions = api.get_position()
            for symbol, order in positions.items():
                if symbol not in code_lst:
                    if order.pos_long > 0:
                        Trd.insert_order_sp_limit(symbol)
                    if order.pos_short > 0:
                        Trd.insert_order_bp_limit(symbol)

            for code in code_lst:
                position_account = api.get_position(code)
                position_long = position_account.pos_long
                position_short = position_account.pos_short
                position = trading_info.loc[code]['position']
                if code == 'DCE.y2009':
                    a = 0
                if position == 0 and position_short == 0 and position_long == 0:
                    print('%s:   仓位%s手, 状态：%s' % (code, position, '完成'))
                    continue
                elif position == position_long and position_short == 0:
                    print('%s: 多头持仓%s手, 状态：%s' % (code, position, '完成'))
                    continue
                elif position == -position_short and position_long == 0:
                    print('%s: 空头持仓%s手, 状态：%s' % (code, position, '完成'))
                    continue
                else:
                    print('%s:   仓位%s手, 状态：%s' % (code, position, '未完成'))
                quote = api.get_quote(code)
                if position > 0:
                    if position_short > 0:
                        order_bp = Trd.insert_order_bp_limit(code)
                    diff = position - position_long
                    if diff > 0:
                        order = Trd.insert_order_bk_limit(code, int(diff))
                    elif diff < 0:
                        order = Trd.insert_order_sp_limit(code, -int(diff))
                if position < 0:
                    if position_long > 0:
                        order_sp = Trd.insert_order_sp_limit(code)
                    diff = -position - position_short
                    if diff > 0:
                        order = Trd.insert_order_sk_limit(code, int(diff))
                    elif diff < 0:
                        order = Trd.insert_order_bp_limit(code, -int(diff))
                if position == 0:
                    if position_short > 0:
                        order_bp = Trd.insert_order_bp_limit(code)
                    if position_long > 0:
                        order_sp = Trd.insert_order_sp_limit(code)
            t_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[-8:]


            time.sleep(60)
