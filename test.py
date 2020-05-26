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


if __name__ == '__main__':
    code = '000001.XSHE'
    eday = datetime.datetime.today()
    SZ50_stocks_list = get_index_stocks('000300.XSHG')
    SZ50_stocks_list = normalize_code(SZ50_stocks_list)
    for code in SZ50_stocks_list:
        a = values_data_cgo(code, 2, eday)


