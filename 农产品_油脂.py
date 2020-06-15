# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pymongo
import datetime
import copy
import math
from arctic import Arctic, TICK_STORE, CHUNK_STORE
style.use('ggplot')
from jqdatasdk import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import copy
auth('18610039264', 'zg19491001')
import datetime
import talib as tb
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from smtplib import SMTP_SSL


def values_data_cgo(stockcode, count, eday):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe 市值表 字段为 code :股票代码  day:日期  capitalization:总股本（万股）
    circulating_cap ：流通股本（万股） market_cap：总市值（亿） circulating_market_cap：流通市值（亿）
    turnover_ratio：换手率 pe_ratio：市盈率 TTM pe_ratio_lyr：静态市盈率  pb_ratio：市净率
    ps_ratio：市销率  pcf_ratio：市现率
    """
    q = query(valuation.code,
              valuation.turnover_ratio,
              ).filter(valuation.code == stockcode)

    df = get_fundamentals_continuously(q, count=count, end_date=eday, panel=False)[
        ['day', 'code', 'turnover_ratio']]

    today = datetime.date.today()
    temp = get_price(stockcode, start_date=today, end_date=today, frequency='daily', fields=None, skip_paused=True,
                     fq='post', count=None).reset_index() \
        .rename(columns={'index': 'day'})
    volume = temp.volume.tolist()[-1] * 100 * 100
    q1 = query(finance.STK_CAPITAL_CHANGE.code,
               finance.STK_CAPITAL_CHANGE.change_date,
               finance.STK_CAPITAL_CHANGE.share_trade_total).filter(finance.STK_CAPITAL_CHANGE.code == stockcode)
    circulating_cap = finance.run_query(q1).sort_values(['change_date']).share_trade_total.tolist()[-1] * 10000
    df_today = pd.DataFrame({'day': [today], 'code': [stockcode], 'turnover_ratio': [volume/circulating_cap]})
    ret = []
    ret.append(df[['day', 'code', 'turnover_ratio']])
    ret.append(df_today)
    ret = pd.concat(ret)
    return ret


def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                     count=None).reset_index() \
        .rename(columns={'index': 'date_time'}) \
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
    return temp


if __name__ == '__main__':
    code_lst = ['Y1409', 'P1409', 'Y1509', 'P1509', 'Y1609', 'P1609', 'Y1709', 'P1709', 'Y1809', 'P1809', 'Y1909', 'P1909', 'Y2009', 'P2009']
    code_lst = [i + '.XDCE' for i in code_lst]
    couple_lst = []
    for i in range(0, len(code_lst), 2):
        couple_lst.append((code_lst[i], code_lst[i+1]))
    s_date = '2013-09-01'
    e_date = '2020-06-01'
    ret = []
    df_all = []
    for (code09, code01) in couple_lst:
        s_date = '20' + str(int(code09[1:3]) - 1) + '-' + '09-01'
        e_date = '20' + str(int(code09[1:3])) + '-' + '10-01'
        hq09 = stock_price(code09, s_date, e_date).assign(close09=lambda df: df.close)[['date_time', 'close09']]
        hq01 = stock_price(code01, s_date, e_date).assign(close01=lambda df: df.close)[['date_time', 'close01']]
        diff = hq09.merge(hq01, on=['date_time']).assign(date=lambda df: df.date_time.apply(lambda x: x[5:]))
        diff['date' + code09[1:3]] = diff['date_time'].apply(lambda x: x[5:])
        diff[code09[1:5]] = diff['close09'] - diff['close01']
        ret.append(diff[['date' + code09[1:3], code09[1:5]]])
        df_all.append(diff[['date_time', 'close09', 'close01']])
    ret = pd.concat(ret, axis=1)[['date16', '2009', '1909', '1809', '1709', '1609', '1509', '1409']].rename(
        columns={'date16': 'date_time'})
    print(ret)

    ret = ret.set_index(['date_time'])
    ret.to_csv('G://zf//YM_01.csv')
    ret.ix[:, ['2009', '1909', '1809', '1709', '1609', '1509', '1409']].plot()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('豆油09-棕榈油09')
    plt.show()

    df_all = pd.concat(df_all).rename(columns={'close09': 'Y09', 'close01': 'P09'})
    exam_X = df_all.loc[:, 'Y09']
    exam_y = df_all.loc[:, 'P09']
    rDF = df_all.corr()
    print(rDF)
    X_train, X_test, y_train, y_test = train_test_split(exam_X, exam_y, train_size=.95, test_size=.05)
    X_train = np.array(X_train)

    df_all.to_csv('G://zf//YM_all.csv')
    model = LinearRegression()
    # 训练模型

    X_train = X_train.reshape(-1, 1)

    model.fit(X_train, y_train)
    a = model.intercept_
    b = model.coef_
    print('最佳拟合线：截距a=', a, '，回归系数b=', b)

    plt.scatter(X_train, y_train, color='r', label='相关系数0.92')  # 绘制散点图
    y_train_pred = model.predict(X_train)  # 训练数据的预测值
    plt.plot(X_train, y_train_pred, color='black', linewidth=3)  # 绘制最佳拟合线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend(loc=2)
    plt.xlabel("豆油")
    plt.ylabel("棕榈油")
    plt.title('豆棕价格线性回归')
    plt.show()



