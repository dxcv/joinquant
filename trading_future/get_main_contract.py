# coding=utf-8
from __future__ import division
import pandas as pd
import datetime
from trading_future.future_singleton import Future
from jqdatasdk import *
auth('15658001226', 'taiyi123')


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour >= 15:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


if __name__ == '__main__':
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                   'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    symbol_lst = ['RU', 'C', 'P', 'AU', 'CU', 'ZN', 'SC', 'Y', 'CF']
    date = datetime.date.today()

    # bars = 5
    # calen = get_trade_days(count=bars)
    # calen = list(calen)
    # calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, date)
    # porfolio = Future()
    # print(EndDate)
    # df = porfolio.get_main_symbol(product=symbol_lst, date=EndDate)
    # df = pd.DataFrame(df).T
    # print(df)
    # df.to_csv('G:/trading/main_contract.csv')


    code = 'Y8888.XDCE'
    sday = '2010-01-01'
    eday = '2020-06-02'
    temp = get_price(code, start_date=sday, end_date=eday, frequency='1m', fields=None, skip_paused=True, fq='pre',
                     count=None)[['open', 'high', 'low', 'close', 'volume']]
    print(temp)
    # temp['date_time'] = temp.index
    # temp['date_time'] = temp['date_time'].apply(lambda x: str(x) + str(' 00:00'))
    # temp = temp.set_index(['date_time'])
    temp.to_csv('e:/data/Y1m.txt')




