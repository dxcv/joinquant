# -*- coding: utf-8 -*-
# @Time    : 2020/6/19 9:56
# @Author  : zhangfang
import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv('g:/trading_strategy/trading_record/' + 'out20200616.txt', encoding='gbk')
    print(df1)
    print(df1.columns)
    print(df1[[' 成交日期']])



