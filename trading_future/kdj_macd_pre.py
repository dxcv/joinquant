# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
# from backtest_func import *
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib import style
import time
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import numpy as np
import talib
import tkinter
import tkinter.messagebox

style.use('ggplot')

auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)


# 获取价格
def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    q = query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == sec).order_by(
        finance.GLOBAL_IDX_DAILY.day.desc())
    temp = finance.run_query(q)[
        ['day', 'name', 'code', 'open', 'high', 'low', 'close', 'volume']] \
        .assign(day=lambda df: df.day.apply(lambda x: str(x)[:10])) \
        .rename(columns={'day': 'trade_date', 'code': 'stock_cpde'})
    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].sort_values(['trade_date'])
    return temp


def get_stock_code_list():
    db_index = jzmongo['stock_raw.stock_index']
    stock_df = db_index.read('all')
    code_list = list(stock_df.iloc[-1].dropna().index)
    return code_list


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                     count=None).reset_index() \
        .rename(columns={'index': 'date_time'}) \
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
    temp['stock_code'] = sec
    return temp


def trans_heng_float(x):
    if x == '--':
        x = None
    return x


def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.stock_1d_jq_post'].read(sec)
    temp = temp[temp['volume'] > 0]
    temp['date_time'] = temp.index
    temp = temp.assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x))) \
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x))) \
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[
        ['high', 'open', 'low', 'close', 'date_time']].dropna()
    temp = temp[(temp['date_time'] >= sday) & (temp['date_time'] <= eday)].sort_values(['date_time'])

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
    return temp


def KDJ(data, N=9, M1=3, M2=3):
    datelen = len(data)
    data = data[['date_time', 'open', 'high', 'low', 'close']]
    array = np.array(data)
    kdjarr = []
    k_lst = []
    d_lst = []
    j_lst = []

    for i in range(datelen):
        if i - N < 0:
            b = 0
        else:
            b = i - N + 1
        rsvarr = array[b:i + 1, 0:5]
        rsv = (float(rsvarr[-1, -1]) - float(min(rsvarr[:, 3]))) / (
                float(max(rsvarr[:, 2])) - float(min(rsvarr[:, 3]))) * 100
        if i == 0:
            k = rsv
            d = rsv
        else:
            k = 1 / float(M1) * rsv + (float(M1) - 1) / M1 * float(kdjarr[-1][2])
            d = 1 / float(M2) * k + (float(M2) - 1) / M2 * float(kdjarr[-1][3])
        j = 3 * k - 2 * d
        k_lst.append(k)
        d_lst.append(d)
        j_lst.append(j)
        kdjarr.append(list((rsvarr[-1, 0], rsv, k, d, j)))

    return k_lst, d_lst, j_lst


def get_alert_info(df, txt):
    if len(df) > 0:
        info_txt = txt
        for idx, row in df_day_buy.iterrows():
            info_txt = info_txt + row.stock_code[:6] + ': k ' + str(np.around(row.k_day, 2)) + ' d ' + \
                       str(np.around(row.d_day, 2)) + ';'
        print(info_txt)
        tkinter.messagebox.showinfo('提示', info_txt)


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold = 'e:/kdj_macd/'
    fold_data = 'e:/kdj_macd/data/'

    start_day = '2017-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    index_code_lst = get_stock_code_list()
    index_code_lst = ['000002.XSHE', '600048.XSHG', '600018.XSHG', '600054.XSHG', '600138.XSHG', '600977.XSHG',
                      '601111.XSHG', '002739.XSHE', '600887.XSHG', '600009.XSHG', '600585.XSHG', '000333.XSHE']
    percentile_lst = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n = 1  # 回测周期
    period = '1d'
    method_lst = [('day', 'day'), ('week', 'day'), ('day', 'week'), ('week', 'week')]
    method_lst = [('week', 'week')]
    k1_lst = [(20, 30), (20, 40), (15, 35), (20, 35), (15, 40)]  # kd下限
    k2_lst = [(70, 80), (60, 80), (65, 85), (65, 80), (60, 85)]  # kd上限
    percentile_lst = [1]
    k1_lst = [(20, 30)]  # kd下限
    k2_lst = [(70, 80)]  # kd上限
    k1 = (0, 100)
    k2 = (0, 100)

    lever_lst = [1]  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever

    fee = 0.00011
    date_lst = [('2008-01-01', '2009-12-31'), ('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'),
                ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    date_lst = [('2015-01-01', '2019-12-31')]
    df_lst = []
    lst = []
    state_lst = []
    signal_state_all_lst = []
    for symble in index_code_lst:
        data_daily = stock_price_cgo(symble, start_day, end_day)[
            ['date_time', 'open', 'high', 'low', 'close', 'stock_code']]
        data_daily['time'] = pd.to_datetime(data_daily['date_time'])
        data_daily.index = data_daily['time']
        data_daily = data_daily.drop(['time'], axis=1)
        data_week = data_daily.resample('W').last()
        data_week['date_time'] = data_daily['date_time'].resample('W').last()
        data_week['open'] = data_daily['open'].resample('W').first()
        data_week['high'] = data_daily['high'].resample('W').max()
        data_week['low'] = data_daily['low'].resample('W').min()
        data_week['close'] = data_daily['close'].resample('W').last()
        data_week = data_week.dropna()
        data_week['k_week'], data_week['d_week'], data_week['j_week'] = KDJ(data_week, 9, 3, 3)

        # data_week['k_week'], data_week['d_week'] = talib.STOCH(
        #     data_week['high'].values, data_week['low'].values, data_week['close'].values, fastk_period=9,
        #     slowk_period=3, slowk_matype=2, slowd_period=3, slowd_matype=2)
        data_week['MACD'], data_week['MACDsignal'], data_week['macd_week'] = talib.MACDEXT(
            data_week['close'].values, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9,
            signalmatype=1)
        data_week = data_week.drop(['MACD', 'MACDsignal'], axis=1)
        data_daily['k_day'], data_daily['d_day'], data_daily['j_day'] = KDJ(data_daily, 9, 3, 3)
        # data_daily['k_day'], data_daily['d_day'] = talib.STOCH(
        #     data_daily['high'].values, data_daily['low'].values, data_daily['close'].values, fastk_period=9,
        #     slowk_period=3, slowk_matype=2, slowd_period=3, slowd_matype=2)
        data_daily['MACD'], data_daily['MACDsignal'], data_daily['macd_day'] = talib.MACDEXT(
            data_daily['close'].values, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9,
            signalmatype=1)
        data_daily = data_daily.drop(['MACD', 'MACDsignal'], axis=1) \
            .merge(data_week[['k_week', 'd_week', 'macd_week', 'date_time']], on=['date_time'], how='left').sort_values(
            ['date_time'])
        data_daily = data_daily.fillna(method='ffill')
        # data_daily.to_csv(fold_data + symble + '_data_daily.csv')
        data_daily['day_kd_b'] = (data_daily['k_day'] > data_daily['d_day']) & (
                data_daily['k_day'].shift(1) < data_daily['d_day'].shift(1)) & (
                                         data_daily['k_day'] < k1[1]) & (data_daily['d_day'] > k1[0])
        data_daily['day_kd_s'] = (data_daily['k_day'] < data_daily['d_day']) & (
                data_daily['k_day'].shift(1) > data_daily['d_day'].shift(1)) & (
                                         data_daily['k_day'] > k2[0]) & (data_daily['d_day'] < k2[1])
        data_daily['week_kd_b'] = (data_daily['k_week'] > data_daily['d_week']) & (
                data_daily['k_week'].shift(1) < data_daily['d_week'].shift(1)) & (
                                          data_daily['k_week'] < k1[1]) & (data_daily['d_week'] > k1[0])
        data_daily['week_kd_s'] = (data_daily['k_week'] < data_daily['d_week']) & (
                data_daily['k_week'].shift(1) > data_daily['d_week'].shift(1)) & (
                                          data_daily['k_week'] > k2[0]) & (data_daily['d_week'] < k2[1])
        df_lst.append(data_daily.tail(1))
    df = pd.concat(df_lst)

    name_lst = []
    for code in df.stock_code.tolist():
        name_lst.append(get_security_info(code).display_name)
    df['股票简称'] = name_lst
    df[['k_day', 'd_day', 'k_week', 'd_week']] = df[['k_day', 'd_day', 'k_week', 'd_week']].apply(lambda x: np.around(x, 2))
    df = df[['date_time', '股票简称', 'stock_code', 'day_kd_b', 'day_kd_s', 'week_kd_b', 'week_kd_s', 'k_day', 'd_day', 'k_week',
             'd_week']].set_index(['股票简称'])

    df_day_buy = df[df['day_kd_b']==True]
    df_day_sell = df[df['day_kd_s']==True]
    df_week_buy = df[df['week_kd_b']==True]
    df_week_sell = df[df['week_kd_s']==True]
    # get_alert_info(df_day_buy, '日级别KD金叉:')
    # get_alert_info(df_week_buy, '周级别KD金叉:')
    # get_alert_info(df_day_sell, '日级别KD死叉:')
    # get_alert_info(df_week_sell, '周级别KD死叉:')

    df.columns = ['日期', '股票代码', '日金叉', '日死叉', '周金叉', '周死叉', '日K', '日D', '周K', '周D']
    print(df[['日金叉', '日死叉', '周金叉', '周死叉', '日K', '日D', '周K', '周D']])

    df.to_csv(fold_data + 'kdj_' + end_day + '.csv', encoding='gbk')
