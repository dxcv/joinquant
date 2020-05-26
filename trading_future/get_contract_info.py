# coding=utf-8
import pandas as pd
import re
import datetime
from trading_future.future_singleton import Future
import pymongo


if __name__ == '__main__':
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                   'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    info_lst = ['ExchangeID', 'VolumeMultiple', 'MaxMarketOrderVolume', 'MaxLimitOrderVolume']  # 'VolumeMultiple',
    contract_lst = ['RU2009', 'FG2009', 'AU2012', 'SR2009', 'P2009', 'C2009', 'ZN2007', 'CU2007', 'SC2007', 'SC2012']
    date = datetime.date.today()
    porfolio = Future()
    df = porfolio.get_contract_info(contract_lst=contract_lst, info_lst=info_lst)
    df = pd.DataFrame(df).T
    print(df)
    df.to_csv('G:/trading/exchageID.csv')
    # 获取持仓限制
    limit_position = porfolio.get_limit_position(contract_lst)
    limit_position_ret = {}
    limit_position_ret['limit_position'] = limit_position
    limit_position = pd.DataFrame(limit_position_ret)
    print(limit_position)