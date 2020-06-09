# %% md


# %%

import pandas as pd
import numpy as np
# import quandl   # 获取股票数据
# !/usr/bin/env python
# coding: utf-8

# 计算
import arctic
import pymongo
import pandas as pd
import datetime as dt
import numpy as np
import math
import statsmodels.api as sm
import logging
import time
import pickle
import os
from copy import deepcopy
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import time
import traceback
import os
from multiprocessing import Pool
import time
from multiprocessing import Process

myclient = pymongo.MongoClient('mongodb://dbmanager_stock:jz471042@223.93.165.111:27017/')
jzmongo = Arctic(myclient)
jzmongo.list_libraries()

from datetime import date
import matplotlib.pyplot as plt


# 累积收益曲线绘制函数
def cumulative_returns_plot(name_list):
    for name in name_list:
        CumulativeReturns = ((1 + StockReturns[name]).loc['2016-01-01':].cumprod() - 1)
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()


# 股票池
stocklist = \
    ['002271.SZ',
     '002460.SZ',
     '000661.SZ',
     '002242.SZ',
     '300014.SZ',
     '600570.SH',
     '002594.SZ',
     '600161.SH',
     '600763.SH',
     '002439.SZ',
     '601012.SH',
     '600031.SH',
     '603288.SH',
     '000858.SZ',
     '600519.SH',
     '601888.SH',
     '002142.SZ',
     '600036.SH',
     '000338.SZ',
     '000333.SZ',
     '600346.SH',
     '601318.SH',
     '600009.SH',
     '002032.SZ',
     '000651.SZ',
     '600309.SH',
     '600340.SH'
     ]
stocklist = [x.replace('.SZ', '.XSHE').replace('.SH', '.XSHG') for x in stocklist]
code_list = stocklist
code_list2 = [code[:6] for code in stocklist]

# 计算每日收益率，并丢弃缺失值
import datetime

stock_1d = jzmongo['stock_raw.stock_1d_jq_post'].read(stocklist, chunk_range=pd.date_range('2014-01-01',
                                                                                           datetime.datetime.today()))
StockPrices = pd.DataFrame()
for stock in stock_1d.keys():
    print(stock)
    StockPrices[stock] = stock_1d[stock]['close']
print(StockPrices)
StockReturns = StockPrices.pct_change().dropna(how='all')
StockReturns.index.name = 'Date'

# 打印前5行数据
print(StockReturns.head())
StockReturns = StockReturns[StockReturns.columns[:]]
StockReturns.head()

# %%

# 数据库
db = myclient['stock_fundamentals']['Updating']
db2 = myclient['stock_fundamentals']['FINSTM_Forecdt']
price = jzmongo['stock_raw.stock_1d_jq_post'].read('000001.XSHE')
stock_return = StockReturns.copy()
# 财报日替换为真实公布日期
data = pd.DataFrame(db.find({'Stkcd': {'$in': code_list2}, 'Accper': {'$gt': '2018-01-01'}, 'gta_name': '净资产收益率TTM'}))
data2 = data[data['Typrep'] == 'A'][['Accper', 'Stkcd', 'value']]
df2 = pd.DataFrame(db2.find({'Stkcd': {'$in': code_list2}}))
df2 = df2[['Stkcd', 'Accper', 'Actudt']]
df = pd.merge(data2, df2, how='inner', on=['Accper', 'Stkcd'])

# 转换为交易日
all_df = pd.DataFrame()
for code in code_list:
    price_single = price
    price_single['Date'] = price_single.index.strftime('%Y-%m-%d')
    #     print(price_single)
    df_single = df[df['Stkcd'] == code[:6]]
    df_single.index = pd.to_datetime(df_single['Actudt'])
    df_single.drop_duplicates(keep='last', subset=['Actudt'], inplace=True)
    df_single['1年ROE增速'] = df_single['value'].diff(4) / df_single['value'].shift(4)
    df_single['3年ROE增速'] = df_single['value'].diff(12) / df_single['value'].shift(12)
    df_single['5年ROE增速'] = df_single['value'].diff(20) / df_single['value'].shift(20)
    #     print(df_single)
    df_single2 = pd.merge(df_single, price_single[['Date']], how='outer', left_index=True, right_index=True)
    df_single2['Date'] = pd.to_datetime(df_single2.index)
    df_single2.fillna(method='ffill', inplace=True)
    df_single2['value'] = df_single2['value'].shift(1)
    df_single2['week'] = df_single2['Date'].dt.weekday
    df_single2 = df_single2[(df_single2['week'] != 5) & (df_single2['week'] != 6)]
    all_df = pd.concat([all_df, df_single2], axis=0)

all_df2 = all_df.loc['2010/1/1':]
final = pd.pivot_table(all_df2, index=all_df2.index, columns='Stkcd', values='value')
final = pd.pivot_table(all_df2, index=all_df2.index, columns='Stkcd', values='1年ROE增速')
final.columns = [x + '.XSHG' if x[0] == '6' else x + '.XSHE' for x in final.columns]

roe_yoy = final.loc['2020/1/1':]
roe_yoy.columns = [x[:11] for x in roe_yoy.columns]
roe_yoy = roe_yoy[list(stock_return.columns)]

roe_yoy = roe_yoy.sub(roe_yoy.mean(axis=1), axis=0).div(roe_yoy.std(axis=1), axis=0)

# 短周期夏普值
stock_sharpe = stock_return.rolling(20).mean() * np.sqrt(250) / stock_return.rolling(20).std()
stock_sharpe.columns = [x[:11] for x in stock_sharpe.columns]
stock_sharpe = stock_sharpe[list(stock_return.columns)]
stock_sharpe = stock_sharpe.sub(stock_sharpe.mean(axis=1), axis=0).div(stock_sharpe.std(axis=1), axis=0)
stock_ret = stock_return.rolling(20).mean() * np.sqrt(250) / stock_return.rolling(20).std()
stock_sharpe = stock_sharpe + roe_yoy
# 4.8 0.8
# 交易日收盘前调仓

# %%

stock_sharpe.tail()

# %%

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

solvers.options['maxiters'] = 8
result = pd.DataFrame()
days = list(stock_sharpe.dropna().index)
position = pd.DataFrame(index=days, columns=stock_sharpe.columns)
# upper个股最高限
# lower个股最低限
upper = 3.4
lower = 0.3

print(upper, lower, 'ing=================================================================')
for date in days[-1:]:
    ret_corr = stock_return.loc[days[days.index(date) - 20]:date]
    try:
        if True:
            ret_corr = stock_ret.loc[days[days.index(date) - 20]:date]
            a = pd.DataFrame(index=range(72)[1:], columns=range(28)[1:])
            for k in range(28)[1:]:
                a.loc[k, k] = 1.
            a.loc[28, 1] = 1.
            a.loc[29, 2] = 1.
            a.loc[30, [3, 8, 9]] = 1.
            a.loc[31, [4, 20, 24, 25]] = 1.
            a.loc[32, 5] = 1.
            a.loc[33, [6, 10]] = 1.
            a.loc[34, [7, 19]] = 1.
            a.loc[35, 11] = 1.
            a.loc[36, 12] = 1.
            a.loc[37, 13] = 1.
            a.loc[38, [14, 15]] = 1.
            a.loc[39, 16] = 1.
            a.loc[40, [17, 18]] = 1.
            a.loc[41, [21, 26]] = 1.
            a.loc[42, 22] = 1.
            a.loc[43, 23] = 1.
            a.loc[44, 27] = 1.
            for k in range(72)[45:]:
                a.loc[k, k - 44] = -1.


            G = matrix(np.array(a.fillna(0.0)))
            # A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
            h = matrix([1. / 27 * upper] * 27 + [1. / 17 * 1.6] * 17 + [-1. / 27 * lower] * 27)
            p = matrix(list(stock_sharpe.loc[date] * (-1)))
            #                     Q = matrix([[0.]*len(stock_sharpe.loc[date])]*27)
            Q = matrix(np.array(2 * ret_corr.corr()))
            A = matrix([1.0] * 27, (1, 27))
            b = matrix(1.0)
            #                     sol = solvers.lp(c,A,b)
            sol = solvers.qp(Q, p, G, h, A, b)
            position.loc[date] = list(sol['x'])


    except Exception as e:

        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        # print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        pass

print(position.iloc[-1].sort_values())



