# coding=utf-8
from __future__ import division
import pandas as pd
import datetime
from trading_future.future_singleton import Future
from jqdatasdk import *
from tqsdk import TqApi, TqAccount
auth('15658001226', 'taiyi123')


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour >= 15:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, StartDate, str(hq_last_date)[:10]


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    # code_lst = temp.symbol.tolist()
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic


def stock_price(sec, sday, eday, fred):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """

    temp = get_price(sec, start_date=sday, end_date=eday, frequency=fred, fields=None, skip_paused=True, fq='pre',
                     count=None)[['open', 'high', 'low', 'close', 'volume']]

    if fred == 'daily':
        temp['date_time'] = temp.index
        temp['date_time'] = temp['date_time'].apply(lambda x: str(x) + str(' 00:00'))
        temp = temp.set_index(['date_time'])
    return temp


if __name__ == '__main__':
    api = TqApi(TqAccount("simnow", "163919", "yo193846"), web_gui=False)
    calen = get_trade_days(count=5)
    today = datetime.date.today()
    calen = list(calen)
    code_dic = get_normal_future_index_code()
    symbol_lst = ['C', 'CS', 'A', 'B', 'M', 'RM', 'Y', 'P', 'OI', 'L', 'V', 'PP', 'TA', 'RU', 'BU', 'MA', 'SC', 'FU',
                   'AL', 'ZN', 'CU', 'PB', 'NI', 'SN', 'J', 'JM', 'I', 'RB', 'HC', 'ZC', 'SF', 'SM', 'FG', 'IF',
                   'IH', 'IC', 'T', 'TF', 'AG', 'AU', 'JD', 'AP', 'CJ', 'CF', 'SR']
    # symbol_lst = ['RU', 'C', 'P', 'AU', 'CU', 'ZN', 'SC', 'Y', 'CF']
    calen, next_tradeday, EndDate, StartDate, hq_last_date = get_date(calen, today)
    # close_dict = {}
    # for index_code in symbol_lst:
    #     code = code_dic[index_code]
    #     index_hq = stock_price(code, hq_last_date, hq_last_date, 'daily')
    #     close_dict[index_code] = index_hq.close.tolist()[-1]
    # print(close_dict)

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
        trading_code = ExchangeID_dict[main_contract]['ExchangeID'] + '.' + ExchangeInstID_dict[main_contract][
            'ExchangeInstID']
        signal_dict[symbol] = {
            'symbol': symbol, 'trading_code': trading_code,
            'last_price': api.get_quote(trading_code).pre_close,
            'VolumeMultiple': VolumeMultiple_dict[main_contract]['VolumeMultiple']
        }
    trading_info = pd.DataFrame(signal_dict).T
    trading_info['market_value'] = trading_info['VolumeMultiple'] * trading_info['last_price']
    trading_info.to_csv('F:/data/future/' + 'future_market_value_' + hq_last_date + '.csv')

    print(trading_info)




