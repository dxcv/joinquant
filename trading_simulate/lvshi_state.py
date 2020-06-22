# -*- coding: utf-8 -*-
# @Time    : 2020/6/19 9:56
# @Author  : zhangfang
import pandas as pd
from backtest_func import *

if __name__ == "__main__":
    df1 = pd.read_csv('g:/trading_strategy/trading_record/' + 'inout20200616.txt.txt', encoding='gbk')
    print(df1)
    print(df1.columns)
    print(df1)
    columns_name = ['成交日期', 'symbol', '买卖', '开平', '成交价', 'volume', 'value', 'fee', '平仓盈亏', '成交时间']
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    print(today)

    f = open('g:/trading_strategy/trading_record/' + 'inout20200616.txt.txt', 'r')
    lst = []
    for name in ['202006161', '202006171', '202006181', '20200619', '202006191', '20200616', '20200617', '20200618', '20200615']:
        with open('g:/trading_strategy/trading_record/' + 'inout' + name + '.txt.txt', "r") as filestream:
            W = False
            for line in filestream:
                new_line = line.split("|")
                print(new_line)
                if len(new_line) > 1:
                    if '成交日期' in new_line[1]:
                        print('True')
                        W = True
                        lenth = len(new_line)
                if (W == True) and (len(new_line) == lenth) and ('成交日期' not in new_line[1]) and (
                        '共' not in new_line[1]):
                    row = []
                    for i in range(1, len(new_line) - 3):
                        row.append(new_line[i].strip())
                    lst.append(row)
    df = pd.DataFrame(lst, columns=columns_name)
    df['平仓盈亏'] = df['平仓盈亏'].apply(lambda x: float(x))
    df['fee'] = df['fee'].apply(lambda x: float(x))
    df['扣除手续费盈亏'] = df['平仓盈亏'] - 2 * df['fee']
    print(df)

    df_sell = df[df['开平'] == '平仓']
    print(df_sell)
    win_R, odds, ave, mid_ret = get_winR_odds(df_sell['扣除手续费盈亏'].tolist())
    df_symbol_state = df_sell[['symbol', '扣除手续费盈亏']].groupby(['symbol']).sum().sort_values(['扣除手续费盈亏']).reset_index(drop=False)
    print(df_symbol_state)


    print('总交易次数： %s 次' % len(df_sell))
    print('胜率：%s ' % win_R)
    print('盈亏比：%s ' % odds)
    print('总收益： %s' % df_sell['平仓盈亏'].sum())
    state_df = pd.DataFrame({'总交易次数': [len(df_sell)], '胜率': [win_R], '盈亏比': [odds],
                             '总收益': [df_sell['平仓盈亏'].sum()], '盈利最大品种': [df_symbol_state.symbol.tolist()[-1]],
                             '盈利最小品种': [df_symbol_state.symbol.tolist()[0]]})
    print(state_df)
    state_df.to_csv('g:/trading_strategy/trading_record/state_' + today + '.csv', encoding='gbk')



    # with open('g:/trading_strategy/trading_record/' + 'inout20200616.txt.txt', 'r+', encoding='gbk') as f:
    #     s = [i[:-1].split('|') for i in f.readlines()]
    #     print(s)
