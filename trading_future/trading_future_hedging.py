from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import copy
import talib as tb

# auth('18610039264', 'zg19491001')
style.use('ggplot')

auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)
from data_engine.data_factory import DataFactory
from data_engine.instrument.future import Future
import data_engine.setting as setting


def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                     count=None).reset_index() \
        .rename(columns={'index': 'date_time'}) \
        .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
    temp['stock_code'] = sec
    return temp


def get_normal_future_contract_code(code_lst):
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-5])
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dict = {}
    for code in code_lst:

        code_dict[code] = temp.loc[code]['index_code']

    return code_dict


if __name__ == '__main__':
    DataFactory.config(MONGDB_PW='jz2018*', DATASOURCE_DEFAULT=setting.DATASOURCE_REMOTE)
    fold = 'e:/kdj_macd/'
    fold_data = 'e:/kdj_macd/data/'
    long_margin = 100000
    short_margin = 100000
    level = 10
    long_value = long_margin * level
    short_value = short_margin * level

    start_day = '2020-03-27'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    long_code_lst = ['SC2006']
    short_code_lst = ['PP2009']
    long_code_dict = get_normal_future_contract_code(long_code_lst)
    short_code_dict = get_normal_future_contract_code(short_code_lst)
    lst = []
    for index_code in long_code_lst:
        symble = long_code_dict[index_code]
        data_daily = stock_price(symble, start_day, end_day)[
            ['date_time', 'open', 'high', 'low', 'close']]
        contract = Future(symbol=index_code[:-4]).contract_size
        price_now = data_daily.close.tolist()[-1]
        long_volume = long_value/len(long_code_lst)/price_now / contract

        row = []
        row.append(index_code)
        row.append(long_volume)
        row.append(price_now)
        row.append(contract)
        lst.append(row)
    for index_code in short_code_lst:
        symble = short_code_dict[index_code]
        data_daily = stock_price(symble, start_day, end_day)[
            ['date_time', 'open', 'high', 'low', 'close']]
        contract = Future(symbol=index_code[:-4]).contract_size
        price_now = data_daily.close.tolist()[-1]
        short_volume = -short_value / len(short_code_lst) / price_now / contract
        row = []
        row.append(index_code)
        row.append(short_volume)
        row.append(price_now)
        row.append(contract)
        lst.append(row)
    df = pd.DataFrame(lst, columns=['symbol', 'volume', 'price', 'contract'])
    df['ratio'] = df.volume/df.volume.tolist()[0]
    df['volume_adj'] = np.around(df['volume'])
    df['value_adj'] = df['volume_adj'] * df['contract'] * df['price']
    print(df)
    print('多头头寸：', df[df['value_adj'] > 0].value_adj.sum())
    print('空头头寸：', df[df['value_adj'] < 0].value_adj.sum())
    print('净头寸：', df.value_adj.sum())



