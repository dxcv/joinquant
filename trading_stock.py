# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 15:15
# @Author  : zhangfang
import pandas as pd

if __name__ == '__main__':
    fold = 'g://trading//股票交易报告//'
    df = pd.read_csv(fold + 'holding.csv', encoding='gbk', index_col=0).set_index(['证券名称'])
    print(df)
    df.loc['总计'] = df.apply(lambda x: x.sum())
    print(df)
    df.to_csv(fold + 'holding_new.csv', encoding='gbk')

    trading_today = pd.read_csv(fold + 'trading_today.csv', encoding='gbk', index_col=0).set_index(['证券名称'])
    trading_today.to_csv(fold + 'trading_today_new.csv', encoding='gbk')