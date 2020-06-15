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
import redis
import json
import threading
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import tkinter
import tkinter.messagebox
from tqsdk import TqApi, TqSim, TqAccount

auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)
from data_engine.data_factory import DataFactory
import data_engine.setting as setting
from email_fuction import send_email

MONGDB_IP = '192.168.2.201'  # 居正公司服务器
MONGDB_USER = 'zhangfang'
MONGDB_PW = 'jz148923'
MONGDB_IP_TICK = '192.168.2.11'  # 居正公司服务器
MONGDB_USER_TICK = 'juzheng'
MONGDB_PW_TICK = 'jz2018*'

REDIS_IP = '192.168.1.36'
REDIS_DB = 0

FUTURE_ORIGINAL_TICK_LIB = 'future_tick_jinshuyuan'
FUTURE_TICK_LIB = 'future_tick'
FUTURE_DAILY_LIB = 'test_1d'
FUTURE_MINUTE_5_LIB = 'test_5m'
FUTURE_MINUTE_LIB = 'test_1m'


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


def get_alert_info(df, txt):
    if len(df) > 0:
        info_txt = txt
        for idx, row in df.iterrows():
            info_txt = info_txt + '净盈亏: ' + str(np.around(row.net_profit, 4)) + '; 多头盈亏: ' + \
                       str(np.around(row.long, 4)) + ' 空头盈亏: ' + \
                       str(np.around(row.short, 4)) + ';'
        print(info_txt)
        tkinter.messagebox.showinfo('提示', info_txt)


class Future:
    """
    全系统Future单例，只需创建一次，合约、品种基础信息类
    """
    def __init__(self):
        self.products = list()
        self.products_base_msg = dict()
        self.products_symbol_msg = dict()
        self.tradedays_msg = dict()
        self.main_contract_msg = dict()
        self.__get_product_mongomsg()
        self.__get_trading_sessions()
        self.__get_tradedays()
        self.__get_main_contract()

    def get_VolumeMultiple(self, contract_lst=None):
        """
        获取合约单位
        """
        info_lst = ['VolumeMultiple']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_base_msg[contract[0:-4]][i] for i in info_lst}
        return dict_all

    def get_ExchangeID(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['ExchangeID']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_ExchangeInstID(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['ExchangeInstID']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_LongMarginRatio(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['LongMarginRatio']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_ShortMarginRatio(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['ShortMarginRatio']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_MaxMarketOrderVolume(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['MaxMarketOrderVolume']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_MaxLimitOrderVolume(self, contract_lst=None):
        """
        获取交易所名称
        """
        info_lst = ['MaxLimitOrderVolume']
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i]
                                  for i in info_lst}
        return dict_all

    def get_contract_info(self, contract_lst=None, info_lst=None):
        """
        获取主力合约
        :return:
        :info_lst:['product', 'symbol', 'ProductID', 'ExchangeID', 'MaxLimitOrderVolume', 'MinLimitOrderVolume',
        'MaxMarketOrderVolume', 'MinMarketOrderVolume', 'LongMarginRatio', 'ShortMarginRatio', 'VolumeMultiple',
        'ExchangeInstID', 'IsTrading']
        """
        dict_all = dict()
        contract_lst = [i.upper() for i in contract_lst]
        for contract in contract_lst:
            dict_all[contract] = {i: self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract][i] for i in info_lst}
            df = self.products_symbol_msg[''.join(re.split(r'[^A-Za-z]', contract))][contract]
            a = 0
        return dict_all

    def __get_product_mongomsg(self):
        """
        获取mongo里的product数据
        :return:
        """
        with pymongo.MongoClient(f'mongodb://{MONGDB_USER}:{MONGDB_PW}@{MONGDB_IP}:27017/') as m_cl:
            col = m_cl['MARKET']['product']
            df_product = pd.DataFrame(col.find({'ProductID': {'$regex': '^[a-zA-Z]{1,2}$'}}))
            # del df_product['contract']
            self.products = list(df_product['ProductID'])
            self.products.remove('OI')
            df_product.index = df_product['ProductID']
            self.products_base_msg = df_product.T.to_dict()
            col = m_cl['MARKET']['instruments']
            del df_product
            df_symbols = pd.DataFrame(col.find({'symbol': {'$regex': '^[a-zA-Z]+[0-9]+$'}, "ASSET_TYPE": 'Future'}))
            df_symbols['product'] = df_symbols['symbol'].str.extract(r'(^[a-zA-Z]+)', expand=False).str.upper()
            for product, symbols in df_symbols.groupby('product'):
                symbols.index = symbols['symbol']
                symbols_dict = symbols.T.to_dict()
                self.products_symbol_msg[product] = symbols_dict

    def __get_main_contract(self):
        """
        获取主力合约
        :return:
        """
        with pymongo.MongoClient(f'mongodb://{MONGDB_USER}:{MONGDB_PW}@{MONGDB_IP}:27017/') as m_cl:
            db = m_cl['MARKET']
            for mark in['', '_OI', '_VOL']:
                mark = 'main_contract' + mark
                col = db[mark]
                df = pd.DataFrame(col.find({}, {'_id': 0}))
                df = df.set_index('date').sort_index()
                df.index = pd.to_datetime(df.index)
                self.main_contract_msg[mark] = df

    def get_windy_code(self, code):
        """
        其他code转windy的code
        :param code: 需要转的code
        :return:
        """
        change_dict = {
            "DCE": 'DCE',
            "CZCE": 'CZC',
            "SHFE": 'SHF',
            "INE": 'INE',
            "CFFEX": 'CFE'
        }
        pattern = re.compile(r'^[a-zA-Z]{1,2}')
        product = (pattern.match(code)).group(0).upper()
        exchange_id = self.products_base_msg[product]['ExchangeID']
        if exchange_id is np.NaN:
            return
        windy_code = code + '.' + change_dict[exchange_id]
        return windy_code

    def get_jq_code(self, code):
        """
        其他code转jq的code
        :param code: 需要转的code
        :return:
        """
        change_dict = {
            "DCE": 'XDCE',
            "CZCE": 'XZCE',
            "SHFE": 'XSGE',
            "INE": 'XINE',
            "CFFEX": 'CCFX'
        }

    def get_main_symbol(self, product=None, date=None):
        """
        :param product: str 或者list
        :param date:
        :return:
        """
        if product:
            product = product if isinstance(product, list) else [product]
        date = pd.to_datetime(date) if date else pd.to_datetime(datetime.date.today())
        df_list = list()
        for mark, df in self.main_contract_msg.items():
            try:
                df = pd.DataFrame(df.loc[date]).T
                df.index = [mark]
                df_list.append(df)
            except:
                df = pd.DataFrame(index=[mark], columns=product)
                df_list.append(df)

        df_daily = pd.concat(df_list)
        if product:
            df_daily = df_daily.loc[:, product]
        df_dict = df_daily.to_dict()
        return df_dict

    def __get_trading_sessions(self):
        """
        获取期货历史交易时间窗口
        :return:
        """
        with pymongo.MongoClient(f'mongodb://{MONGDB_USER}:{MONGDB_PW}@{MONGDB_IP}:27017/') as m_cl:
            col = m_cl['MARKET']['TradingSessions']
            trading_sessions_df = pd.DataFrame(col.find())
            del trading_sessions_df['_id']
            trading_sessions_df['DateRange_Start'] = trading_sessions_df['DateRange_Start'].fillna('1990/01/01')
            trading_sessions_df['DateRange_End'] = trading_sessions_df['DateRange_End'].fillna('2099/01/01')
            for product in self.products_base_msg.keys():
                product_trading_sessions = trading_sessions_df.loc[trading_sessions_df['Market'] == product]
                self.products_base_msg[product]['trading_session'] = product_trading_sessions

    def get_product_trading_sessions(self, product, date: str = None):
        """
        获取交易的时间窗口
        :param product:  品种
        :param date: 日期, 默认今日，如果是'all'，返回全周期的
        :return:
        """
        trade_sessions = self.products_base_msg[product]['trading_session']
        if date != 'all':
            date = pd.to_datetime(date) if date else datetime.date.today()
            date = date.strftime('%Y/%m/%d')

            trade_sessions = trade_sessions.loc[(trade_sessions['DateRange_Start'] <= date) &
                                                (trade_sessions['DateRange_End'] >= date), ]
            trade_sessions = trade_sessions
        return trade_sessions

    def __get_tradedays(self):
        change_dict = {
            "DCE": 'DCE',
            "CZCE": 'CZC',
            "SHFE": 'SHF',
            "INE": 'SHF',
            "CFFEX": 'CFE'
        }

        with pymongo.MongoClient(f'mongodb://{MONGDB_USER}:{MONGDB_PW}@{MONGDB_IP}:27017/') as m_cl:
            db = m_cl['Tradedays']
            for jz_code, mongo_code in change_dict.items():
                col = db[mongo_code]
                tradedays_df = pd.DataFrame(col.find({'isTradingday': True}))
                tradedays_df = tradedays_df[['Tradedays_str', 'Tradedays']]
                tradedays_df['Tradedays'] = pd.to_datetime(tradedays_df['Tradedays'].dt.strftime("%Y-%m-%d %H:%M:%S"))
                tradedays_df.drop_duplicates(subset=['Tradedays_str'], inplace=True)
                tradedays_df.set_index('Tradedays', inplace=True)
                tradedays_df.sort_index(inplace=True)
                self.tradedays_msg[jz_code] = tradedays_df

    def find_tradeday(self, day: int, date=None, exchange: str = 'DCE'):
        """
        根据date查询距此日day天的交易日
        :param date: None 默认是今日
        :param day: day为0 时为判断今天是否是交易日，返回Bool
        :param exchange:
        :return: date:str
        """

        date = pd.to_datetime(datetime.date.today()) if not date else pd.to_datetime(date).replace(hour=0,
                                                                                                   minute=0, second=0)
        tradeday_df = self.tradedays_msg[exchange]
        if day == 0:
            return date.strftime("%Y-%m-%d") in tradeday_df['Tradedays_str'].values
        if day > 0:
            tradeday_df = tradeday_df.loc[tradeday_df.index > date]
            return tradeday_df.iloc[day-1]['Tradedays_str']
        if day < 0:
            tradeday_df = tradeday_df.loc[tradeday_df.index < date]
            return tradeday_df.iloc[day]['Tradedays_str']

    def get_limit_position(self, symbols):
        """
        获取合约今日的最大持仓限制
        :param symbol: jz格式
        :return:
        """
        symbols = symbols if isinstance(symbols, list) else [symbols]
        data_dict = dict()
        for symbol in symbols:
            pattern = re.compile(r'^[a-zA-Z]{1,2}')
            product = (pattern.match(symbol)).group(0).upper()
            expireDate = self.products_symbol_msg[product][symbol]['ExpireDate']
            exchange_id = self.products_base_msg[product]['ExchangeID']
            today = datetime.date.today().strftime('%Y-%m-%d')
            if pd.to_datetime(today).strftime('%Y%m%d') >= expireDate:
                data_dict[symbol] = 'expired'
            elif product in ['CF', 'SR', 'TA', 'OI', 'MA', 'FG', 'RM', 'ZC', 'PM', 'WH', 'RS', 'RI', 'JR', 'LR', 'SF',
                             'SM', 'CY', 'AP']:
                expireDate = pd.to_datetime(expireDate)
                lst_lst_month_last_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)
                                          ).replace(day=1) - datetime.timedelta(days=1)
                last_change_day = self.find_tradeday(15, date=lst_lst_month_last_day, exchange=exchange_id)  # 交割月前一个月第15个日历日期间的交易日
                change_day = expireDate.replace(day=1).strftime('%Y-%m-%d')
                if product == 'AP' and symbol[-1] == '7':
                    product = product + '7'
                if today <= last_change_day:
                    limit_dict = {
                        'CF': '单边持仓量<15万:15000手，单边持仓量≥15万:单边持仓量×10%',
                        'SR': '单边持仓量<25万:25000手，单边持仓量≥25万:单边持仓量×10%',
                        'TA': '单边持仓量<25万:25000手，单边持仓量≥25万:单边持仓量×10%',
                        'OI': '单边持仓量<10万:10000手，单边持仓量≥10万:单边持仓量×10%',
                        'MA': '单边持仓量<10万:10000手，单边持仓量≥10万:单边持仓量×10%',
                        'FG': '单边持仓量<20万:20000手，单边持仓量≥20万:单边持仓量×10%',
                        'RM': '单边持仓量<20万:20000手，单边持仓量≥20万:单边持仓量×10%',
                        'ZC': '单边持仓量<60万:60000手，单边持仓量≥60万:单边持仓量×10%',
                        'PM': 2000, 'WH': 2500, 'RS': 10000, 'RI': 7500, 'JR': 20000,
                        'LR': 20000, 'SF': 8000, 'SM': 30000, 'CY': 5000, 'AP': 500, 'AP7': 100
                    }

                elif change_day > today > last_change_day:
                    limit_dict = {
                        'CF': 3000, 'SR': 5000, 'TA': 10000, 'OI': 3000, 'MA': 2000, 'FG': 5000,
                        'RM': 2000, 'ZC': 20000, 'PM': 600, 'WH': 1000, 'RS': 1000, 'RI': 2000, 'JR': 3000,
                        'LR': 3000, 'SF': 2000, 'SM': 10000, 'CY': 500, 'AP': 100, 'AP7': 20
                    }
                else:
                    limit_dict = {
                        'CF': 400, 'SR': 1000, 'TA': 5000, 'OI': 1000, 'MA': 1000, 'FG': 1000,
                        'RM': 1000, 'ZC': 4000, 'PM': 200, 'WH': 300, 'RS': 500, 'RI': 400, 'JR': 500,
                        'LR': 500, 'SF': 500, 'SM': 2000, 'CY': 100, 'AP': 10, 'AP7': 6
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['CJ']:
                expireDate = pd.to_datetime(expireDate)
                lst_lst_month_last_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)
                                          ).replace(day=1) - datetime.timedelta(days=1)
                lst_month_fst_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
                lst_lst_change_day = self.find_tradeday(-1, date=lst_month_fst_day, exchange=exchange_id)
                last_change_day = self.find_tradeday(15, date=lst_lst_month_last_day, exchange=exchange_id)
                change_day = expireDate.replace(day=1).strftime('%Y-%m-%d')
                if today <= lst_lst_change_day:
                    limit_dict = {
                        'CJ': 300
                    }

                elif last_change_day >= today > lst_lst_change_day:
                    limit_dict = {
                        'CJ': 60
                    }
                elif change_day > today > last_change_day:
                    limit_dict = {
                        'CJ': 20
                    }
                else:
                    limit_dict = {
                        'CJ': 6
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['A', 'V', 'PP', 'C', 'B', 'L', 'P', 'J', 'JM', 'I', 'FB', 'BB', 'CS', 'Y', 'M', 'EG', 'JD']:
                expireDate = pd.to_datetime(expireDate)
                lst_lst_month_last_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)
                                          ).replace(day=1) - datetime.timedelta(days=1)
                last_change_day = self.find_tradeday(14, date=lst_lst_month_last_day,
                                                     exchange=exchange_id)  # 交割月前一个月第15个日历日期间的交易日
                change_day = self.find_tradeday(-1, date=expireDate.replace(day=1), exchange=exchange_id)
                if today <= last_change_day:
                    limit_dict = {
                        'A': '单边持仓量<=20万:20000手，单边持仓量>20万:单边持仓量×10%',
                        'V': '单边持仓量<=20万:20000手，单边持仓量>20万:单边持仓量×10%',
                        'PP': '单边持仓量<=20万:20000手，单边持仓量>20万:单边持仓量×10%',
                        'C': '单边持仓量<=40万:40000手，单边持仓量>40万:单边持仓量×10%',
                        'B': '单边持仓量<=20万:20000手，单边持仓量>20万:单边持仓量×10%',
                        'L': '单边持仓量<=10万:10000手，单边持仓量>10万:单边持仓量×10%',
                        'P': '单边持仓量<=10万:10000手，单边持仓量>10万:单边持仓量×10%',
                        'J': '单边持仓量<=5万:5000手，单边持仓量>5万:单边持仓量×10%',
                        'JM': '单边持仓量<=8万:8000手，单边持仓量>8万:单边持仓量×10%',
                        'I': '单边持仓量<=40万:40000手，单边持仓量>40万:单边持仓量×10%',
                        'FB': '单边持仓量<=16万:16000手，单边持仓量>16万:单边持仓量×10%',
                        'BB': '单边持仓量<=6万:6000手，单边持仓量>6万:单边持仓量×10%',
                        'CS': '单边持仓量<=15万:15000手，单边持仓量>15万:单边持仓量×10%',
                        'Y': '单边持仓量<=20万:20000手，单边持仓量>20万:单边持仓量×10%',
                        'M': '单边持仓量<=40万:40000手，单边持仓量>40万:单边持仓量×10%',
                        'EG': '单边持仓量<=8万:8000手，单边持仓量>8万:单边持仓量×10%,'
                              ' 单边持仓＞12万:3000手, 提高保证金合约价值×10%',
                    }

                elif change_day >= today > last_change_day:
                    limit_dict = {
                        'A': 5000, 'V': 5000, 'PP': 5000, 'C': 15000, 'B': 4500, 'L': 3000, 'P': 1500, 'J': 900,
                        'JM': 1500, 'I': 6000, 'FB': 400, 'BB': 80, 'CS': 4500, 'Y': 3000, 'M': 7500,
                        'EG': '3000手,单边持仓＞8万:1000手, 提高保证金合约价值×20% '
                    }
                else:
                    limit_dict = {
                        'A': 2500, 'V': 2500, 'PP': 2500, 'C': 5000, 'B': 1500, 'L': 1000, 'P': 500, 'J': 300,
                        'JM': 500, 'I': 2000, 'FB': 100, 'BB': 20, 'CS': 1500, 'Y': 1000, 'M': 2500,
                        'EG': '1000手,单边持仓＞8万:1000手, 提高保证金合约价值×20% '
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['JD']:
                expireDate = pd.to_datetime(expireDate)
                lst_lst_month_last_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)
                                          ).replace(day=1) - datetime.timedelta(days=1)
                lst_lst_change_day = self.find_tradeday(1, date=lst_lst_month_last_day, exchange=exchange_id)
                last_change_day = self.find_tradeday(10, date=lst_lst_month_last_day, exchange=exchange_id)
                change_day = expireDate.replace(day=1).strftime('%Y-%m-%d')
                if today < lst_lst_change_day:
                    limit_dict = {
                        'JD': 600
                    }

                elif last_change_day > today >= lst_lst_change_day:
                    limit_dict = {
                        'JD': 400
                    }
                elif change_day > today >= last_change_day:
                    limit_dict = {
                        'JD': 120
                    }
                else:
                    limit_dict = {
                        'JD': 20
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'RB', 'WR', 'HC', 'RU', 'BU', 'AU', 'AG', 'SP']:
                expireDate = pd.to_datetime(expireDate)
                lst_month_fst_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
                last_change_day = self.find_tradeday(-1, date=lst_month_fst_day, exchange=exchange_id)  # 合约挂牌至交割月前第二月的最后一个交易日
                change_day = expireDate.replace(day=1).strftime('%Y-%m-%d')
                if today <= last_change_day:
                    limit_dict = {
                        'CU': '单边持仓量<16万:8000, 单边持仓量>=16万:单边持仓量×10%',
                        'AL': '单边持仓量<20万:10000, 单边持仓量>=20万:单边持仓量×10%',
                        'ZN': '单边持仓量<12万:6000, 单边持仓量>=12万:单边持仓量×10%',
                        'PB': '单边持仓量<10万:5000， 单边持仓量>=10万:单边持仓量×10%',
                        'NI': '单边持仓量<12万:6000， 单边持仓量>=12万:单边持仓量×10%',
                        'SN': '单边持仓量<3万:1500， 单边持仓量>=3万:单边持仓量×10%',
                        'RB': '单边持仓量<180万:90000， 单边持仓量>=180万:单边持仓量×10%',
                        'WR': '单边持仓量<45万:22500， 单边持仓量>=45万:单边持仓量×10%',
                        'HC': '单边持仓量<240万:120000， 单边持仓量>=240万:单边持仓量×10%',
                        'RU': 500, 'BU': 8000, 'AU': '非期货公司会员:18000, 客户:9000',
                        'AG': '非期货公司会员:18000手, 客户:9000手', 'SP': 4500
                    }

                elif change_day > today > last_change_day:
                    limit_dict = {
                         'CU': 3000, 'AL': 3000, 'ZN': 2400, 'PB': 1800, 'NI': 1800, 'SN': 600, 'RB': 4500, 'WR': 1800,
                         'HC': 9000, 'RU': 150, 'BU': 1500, 'AU': '非期货公司会员:5400, 客户:2700',
                         'AG': '非期货公司会员:5400手, 客户:2700手', 'SP': 900
                    }
                else:
                    limit_dict = {
                         'CU': 1000, 'AL': 1000, 'ZN': 800, 'PB': 600, 'NI': 600, 'SN': 200, 'RB': 900, 'WR': 360,
                         'HC': 1800, 'RU': 50, 'BU': 500, 'AU': '非期货公司会员:1800手, 客户:900手',
                         'AG': '非期货公司会员:1800手, 客户:900手', 'SP': 300
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['FU', 'SC']:
                expireDate = pd.to_datetime(expireDate)

                last_change_day = ((expireDate.replace(day=1) -
                                    datetime.timedelta(days=1)).replace(day=1)
                                   - datetime.timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
                change_day = (expireDate.replace(day=1) - datetime.timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')

                if today < last_change_day:
                    limit_dict = {
                        'FU': 7500, 'SC': 3000
                    }

                elif change_day > today >= last_change_day:
                    limit_dict = {
                        'FU': 1500, 'SC': 1500
                    }
                else:
                    limit_dict = {
                        'FU': 500, 'SC': 500
                    }
                data_dict[symbol] = limit_dict[product]
            elif product in ['TF', 'TS', 'T']:
                expireDate = pd.to_datetime(expireDate)
                change_day = self.find_tradeday(-1, date=expireDate.replace(day=1), exchange=exchange_id)
                if today < change_day:
                    limit_dict = {
                        'TF': 2000, 'TS': 2000, 'T': 2000
                    }
                else:
                    limit_dict = {
                        'TF': 600, 'TS': 600, 'T': 600
                    }
                data_dict[symbol] = limit_dict[product]
        return data_dict


if __name__ == '__main__':
    # DataFactory.config(MONGDB_PW='jz2018*', MONGDB_IP='192.168.2.201', MONGDB_USER='juzheng',
    #                    DATASOURCE_DEFAULT=global_variable.DATASOURCE_REMOTE
    #                    , logging_level=global_variable.logging.INFO)
    # rd = redis.Redis('192.168.1.36')
    api = TqApi(TqAccount("G国泰君安", "85030120", "jz04282020"), web_gui=True)
    Trd = Trading(api)
    order = Trd.insert_order_sp_limit('INE.sc2007', 1)
    order = Trd.insert_order_bk_limit('INE.sc2008', 1)

