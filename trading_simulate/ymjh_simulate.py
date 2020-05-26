# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 13:39
# @Author  : zhangfang
from __future__ import division
import numpy as np
import pandas as pd
import datetime
from jqdatasdk import *
from trading_future.future_singleton import Future
from ymjh_email_future import stock_price
from tqsdk import TqApi
# auth('18610039264', 'zg19491001')
auth('15658001226', 'taiyi123')


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour > 15:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


if __name__ == '__main__':
    api = TqApi(web_gui=True)
    signal_path = 'G://trading_strategy//'
    aum = 1000000
    balance = 10
    bars = 5
    calen = get_trade_days(count=bars)
    today = datetime.date.today()
    calen = list(calen)
    if today in calen:
        calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, today)
        # EndDate = today
        signal = pd.read_csv(signal_path + 'temp//ymjh_' + hq_last_date + '.csv', index_col=0)
        print(signal)
        signal['symbol'] = signal.index
        symbol_lst = signal.symbol.tolist()
        porfolio = Future()
        main_contract_dict = porfolio.get_main_symbol(product=symbol_lst, date=EndDate)
        main_contract = pd.DataFrame(main_contract_dict).T[['main_contract']]
        print(main_contract)
        contract_lst = main_contract.main_contract.tolist()
        ExchangeID_dict = porfolio.get_ExchangeID(contract_lst=contract_lst)
        ExchangeInstID_dict = porfolio.get_ExchangeInstID(contract_lst=contract_lst)
        VolumeMultiple_dict = porfolio.get_VolumeMultiple(contract_lst)

        main_contract['symbol'] = main_contract.index
        print(main_contract)
        signal_dict = {}
        for symbol in symbol_lst:
            main_contract = main_contract_dict[symbol]['main_contract']
            trading_code = ExchangeID_dict[main_contract]['ExchangeID'] + '.' + ExchangeInstID_dict[main_contract]['ExchangeInstID']
            signal_dict[symbol] = {
                'symbol': symbol, 'trading_code': trading_code, 'weight': signal.loc[symbol]['weight'],
                'last_price': api.get_quote(trading_code).pre_close,
                'VolumeMultiple': VolumeMultiple_dict[main_contract]['VolumeMultiple']
                                   }
        trading_info = pd.DataFrame(signal_dict).T
        trading_info['position'] = aum * balance/len(symbol_lst) * trading_info['weight']/trading_info['last_price']/trading_info['VolumeMultiple']
        trading_info['position'] = trading_info['position'].apply(lambda x: int(np.around(x, 0)))
        trading_info.to_csv(signal_path + 'position_ymjh_' + hq_last_date + '.csv')

        print(trading_info)

