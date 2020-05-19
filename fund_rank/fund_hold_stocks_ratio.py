# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:56:55 2020
将混合型基金 分为偏股混合 偏债混合以及 标准混合型
@author: Administrator
"""

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

# from statsmodels import regression
# import statsmodels.api as sm
from jqdatasdk import *

auth('18610039264', 'zg19491001')


# count_=get_query_count()
# print(count_)

# 提取符合条件的基金名单
def fund_find(start_day, operate_mode, underlying_asset_type):
    q = query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.operate_mode_id == operate_mode,
                                             finance.FUND_MAIN_INFO.underlying_asset_type_id == underlying_asset_type,
                                             finance.FUND_MAIN_INFO.start_date < start_day)
    df = finance.run_query(q)
    print('一共' + str(len(df)) + '只基金')
    return (df)


def cal_hybrid_hold_ratio():
    today = datetime.datetime.today()
    today = str(today)[:10]
    period = 36
    fund_stime = str(datetime.datetime.today() - relativedelta(months=period))[:10]  # 基金池开始时间

    operate_mode_id = [401001, 401003, 401006]
    underlying_asset_type_id = [402004]
    # fund_id 为符合条件的基金名单
    ret = list()
    for i in operate_mode_id:
        for j in underlying_asset_type_id:
            tmp = fund_find(fund_stime, i, j)
            ret.append(tmp)
    fund_id = pd.concat(ret)
    fund_id['end_date'] = fund_id['end_date'].fillna(1)
    fund_id = fund_id[fund_id['end_date'] == 1]
    
    fund_lst = fund_id.main_code.tolist()
    ret = list()
    for fd in fund_lst:
        print(fd + ' is cal')
        q = query(finance.FUND_PORTFOLIO.code,
                  finance.FUND_PORTFOLIO.period_end,
                  finance.FUND_PORTFOLIO.stock_rate).filter(finance.FUND_PORTFOLIO.code==fd).order_by(finance.FUND_PORTFOLIO.period_end.desc()).limit(12)
        ss = finance.run_query(q)
        ret.append(ss)
    fund_hold_ratio = pd.concat(ret)
    out = fund_hold_ratio.groupby(['code'])['stock_rate'].mean().reset_index()
    return out


if __name__ == '__main__':
    hybrid_r = cal_hybrid_hold_ratio()
    hybrid_r['label'] = hybrid_r.stock_rate.apply(lambda s: '偏股型' if s > 70
                                                  else '偏债型' if s < 30 else '标准型')
    print(hybrid_r)
    hybrid_r.to_csv('g://lfp//result//fund_class.csv', encoding='gbk')
    

    
