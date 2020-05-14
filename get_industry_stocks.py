# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
from jqdatasdk import *
from cgo_ymjh_email_index_prediction import index_stocks

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

def transfer_code(x):
    x = str(x)
    if len(x) < 6:
        x = '0' * (6-len(x)) + x
    return x

if __name__ == '__main__':
    today = datetime.datetime.today()
    code = pd.read_csv('G://trading//hold_stock.csv', encoding='gbk')
    code['code'] = code['hold_stock'].apply(lambda x: transfer_code(x))
    print(code)
    code_lst = normalize_code(code.code.tolist())
    print(code_lst)
    lst = []
    for code in code_lst:


        d = get_industry(security=code, date=today)
        print(d)
        df = pd.DataFrame(d).T
        print(df)
        df = df[['sw_l1', 'sw_l2']]
        print(df)
        df['sw_l1'] = df['sw_l1'].apply(lambda x: x['industry_name'])
        df['sw_l2'] = df['sw_l2'].apply(lambda x: x['industry_name'])
        lst.append(df)
    ret = pd.concat(lst)

    # df.loc[:, ['sw_l1', 'sw_l2']] = df.loc[:, ['sw_l1', 'sw_l2']].apply(lambda x: x['industry_name'])
    print(ret)
    ret.to_csv('G://trading//stocks_industry.csv', encoding='gbk')


    # index_code_lst = ['399006.XSHE', '000300.XSHG', '000905.XSHG', '000016.XSHG']
    # for i in range(len(index_code_lst)):
    #     code = index_code_lst[i]
    #     SZ50_stocks_list = index_stocks(code)
    #     SZ50_stocks_list = normalize_code(SZ50_stocks_list)
    #     for code in SZ50_stocks_list:
    #         q = query(finance.STK_CAPITAL_CHANGE.code, valuation.code,
    #           valuation.turnover_ratio,).filter(finance.STK_CAPITAL_CHANGE.code == '600276.XSHG')
    #                                                      # finance.STK_CAPITAL_CHANGE.pub_date > '2019-01-01')
    #         df = finance.run_query(q)
    #         print(df)