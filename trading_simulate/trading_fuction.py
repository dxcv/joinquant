# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 13:26
# @Author  : zhangfang
from tqsdk import TqApi, TqSim, TqAccount
import pandas as pd
import time
from jqdatasdk import *
import datetime
auth('15658001226', 'taiyi123')


class Trading:
    def __init__(self, api):
        self.api = api

    def get_time_allowed(self, time_allowed_lst):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[-8:]
        if now > '00:00:00' and now < '03:00:00':
            now = str(int(now[1]) + 24) + now[2:]
        allowed = False
        for (s_time, e_time) in time_allowed_lst:
            if now > s_time and now < e_time:
                allowed = True
                break
        return allowed

    def get_date(self, calen, today):
        next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
        if datetime.datetime.now().hour >= 15:
            calen.append(next_tradeday)
        EndDate = calen[-1]
        StartDate = calen[0]
        hq_last_date = calen[-2]
        return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]

    def insert_order_bk_limit(self, code, volume):
        quote = self.api.get_quote(code)
        limit_price = quote['upper_limit']
        order = None
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            order = self.api.insert_order(code, direction='BUY', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_sk_limit(self, code, volume):
        quote = self.api.get_quote(code)
        limit_price = quote['lower_limit']
        order = None
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            order = self.api.insert_order(code, direction='SELL', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_bp_limit(self, code, volume=None):
        position_account = self.api.get_position(code)
        position_short = position_account.pos_short
        order = None
        quote = self.api.get_quote(code)
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if position_short:
            if volume:
                position_short = min(position_short, volume)
            quote = self.api.get_quote(code)
            limit_price = quote['upper_limit']
            if limit_price > 0:
                order = self.api.insert_order(code, direction='BUY', offset='CLOSE', volume=position_short,
                                         limit_price=limit_price)
                a = 0
                while (order.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_bpk_limit(self, code, volume):
        position_account = self.api.get_position(code)
        position_short = position_account.pos_short
        order = None
        quote = self.api.get_quote(code)
        limit_price = quote['upper_limit']
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            if position_short:
                order_bp = self.api.insert_order(code, direction='BUY', offset='CLOSE', volume=position_short, limit_price=limit_price)
                a = 0
                while (order_bp.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order_bp.status, order_bp.volume_left))
            order = self.api.insert_order(code, direction='BUY', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_spk_limit(self, code, volume):
        position_account = self.api.get_position(code)
        position_long = position_account.pos_long
        order = None
        quote = self.api.get_quote(code)
        limit_price = quote['lower_limit']
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if limit_price > 0:
            if position_long:
                order_sp = self.api.insert_order(code, direction='SELL', offset='CLOSE', volume=position_long, limit_price=limit_price)
                a = 0
                while (order_sp.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order_sp.status, order_sp.volume_left))
            order = self.api.insert_order(code, direction='SELL', offset='OPEN', volume=volume, limit_price=limit_price)
            a = 0
            while (order.status != "FINISHED") and (a < 20):
                a += 1
                self.api.wait_update()
                print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order

    def insert_order_sp_limit(self, code, volume=None):
        position_account = self.api.get_position(code)
        position_long = position_account.pos_long
        order = None
        quote = self.api.get_quote(code)
        time_allowed_lst = quote.trading_time.day
        time_allowed_lst.extend(quote.trading_time.night)
        allowed = self.get_time_allowed(time_allowed_lst)
        if not allowed:
            print('trading time is not allowed')
            return order
        if position_long:
            if volume:
                position_long = min(position_long, volume)
            quote = self.api.get_quote(code)
            limit_price = quote['lower_limit']
            if limit_price > 0:
                order = self.api.insert_order(code, direction='SELL', offset='CLOSE', volume=position_long,
                                         limit_price=limit_price)
                a = 0
                while (order.status != "FINISHED") and (a < 20):
                    a += 1
                    self.api.wait_update()
                    print("code: %s, 委托单状态: %s, 未成交手数: %d 手" % (code, order.status, order.volume_left))
        return order
