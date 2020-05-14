# coding=utf-8
'''
Created on 7.9, 2018

@author: fang.zhang
'''
from __future__ import division
import numpy as np
import math
from configdb import *
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import time
import sqlalchemy
import os
import talib
import copy
import datetime

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

def transfer_code(x):
    if x == None:
        return x
    x = str(int(x))
    if len(x) < 6:
        x = '0' * (6-len(x)) + x
    return x


class MysqlDBConnector(object):
    def __init__(self, dbKey=None):
        if dbKey is None:
            self.connPara = MssqlConnParaMap['local']
        else:
            self.connPara = MssqlConnParaMap[dbKey]
        return

    def build_database_connection(self):
        try:
            conn = pymysql.connect(host=self.connPara['server'], user=self.connPara['user'],
                                   passwd=self.connPara['password'], db=self.connPara['database'])
        except pymysql.DatabaseError  as e:
            print("Can not connect to server")
        return conn

    def build_alchemy_connection(self):
        try:
            engine = create_engine(
                'mysql+pymysql://' + self.connPara['user'] + ':' + self.connPara['password'] + '@' + self.connPara[
                    'server'] + ':3306/' + self.connPara['database'])
        except engine.Error  as e:
            print("Can not connect to server")
        return engine

    def write_data_to_db(self, datadf, TableName, mode=1):
        engine = self.build_alchemy_connection()
        if mode == 2:
            datadf.to_sql(TableName, engine, if_exists='replace', index=False, index_label=None, chunksize=None,
                          dtype={'data_type': sqlalchemy.types.String(32), 'last_time': sqlalchemy.TIMESTAMP})
        elif mode == 3:
            datadf.to_sql(TableName, engine, if_exists='append', index=False, index_label=None, chunksize=None,
                          dtype=None)
        else:
            datadf.to_sql(TableName, engine, if_exists='fail', index=False, index_label=None, chunksize=None,
                          dtype=None)

    def get_data_from_query(self, stmt):
        conn = self.build_database_connection()
        df = pd.read_sql(stmt, conn)
        conn.close()
        return df

    def delete_data_from_query(self, stmt):
        conn = self.build_database_connection()
        cursor = conn.cursor()
        ret = cursor.execute(stmt)
        cursor.close()
        conn.close()
        return ret

    def get_query_stmt(self, tableName, colNames, constraints, orderby):
        stmt = 'select '
        for col in colNames:
            stmt = stmt + col + ','
        stmt = stmt[0:len(stmt) - 1]
        stmt = stmt + ' from ' + tableName
        if constraints is None:
            stmt = stmt + ''
        else:
            stmt = stmt + constraints

        if orderby is None:
            return stmt
        else:
            stmt = stmt + ' order by '
            stmt = stmt + orderby
            return stmt

    def get_data(self, tableName, colNames, constraints, orderby):
        assert colNames is not None
        try:
            conn = self.build_database_connection()
            stmt = self.get_query_stmt(tableName, colNames, constraints, orderby)
            cursor = conn.cursor()
            t = time.time()
            cursor.execute(stmt)
            df = pd.DataFrame.from_records(cursor.fetchall())
            if len(df) > 0:
                df.columns = colNames
        except pymysql.Error as e:
            conn.rollback()
            message = "SqlServer Error %d: %s" % (e.args[0], e.args[1])
            print(message)
        finally:
            cursor.close()
            conn.close()
        print('time elapsed for this oracle query: ', time.time() - t)
        return df

    def update_data_to_db(self, TableName, chng_cloname, chng_clovalue, cloname_lst, value_lst):
        connect = pymysql.connect(host=self.connPara['server'], port=3306, user=self.connPara['user'],
                                  passwd=self.connPara['password'], db=self.connPara['database'],
                                  charset='utf8')
        cur = connect.cursor()
        sql1 = 'update ' + TableName + ' set ' + chng_cloname + ' = %s where ('
        sql2 = ''
        for colname in cloname_lst:
            sql2 = sql2 + colname + '= %s)&('
        sql = sql1 + sql2[:-2]
        lst = []
        lst.append(chng_clovalue)
        lst.extend(value_lst)
        cur.execute(sql, lst)
        connect.commit()
        connect.close()


def get_symble_id(exchange, codecoin, basecoin):
    stmt = 'select id from sys_exchange where name like ' + '\'' + exchange + '\''
    csd_hq = MysqlDBConnector('production')
    exchange_id = csd_hq.get_data_from_query(stmt).iat[0, 0]
    symbol_stmt = 'select id from sys_exsymbol where exchangeid = ' + '\'' + str(exchange_id) + '\'' + \
                  ' and basecoin like ' + '\'' + codecoin + '\'' + ' and quotecoin like ' + '\'' + basecoin + '\''
    data = csd_hq.get_data_from_query(symbol_stmt)
    return data.iat[0, 0]


def get_symble_id_all(exchange):
    stmt = 'select id, symbol from sys_exchange where name like ' + '\'' + exchange + '\''
    csd_hq = MysqlDBConnector('production')
    exchange_id = csd_hq.get_data_from_query(stmt).iat[0, 0]
    symbol_stmt = 'select id from sys_exsymbol where exchangeid like ' + '\'' + str(exchange_id) + '\''
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(symbol_stmt)
    return data


def get_market_kline(exsymbol_id, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, count, high, low, open, period, tickid, volume  from market_kline where exsymbolid ' \
           '= ' + '\'' + str(exsymbol_id) + '\'' + ' and period like ' + '\'' + period + '\'' + ' and tickid between ' \
           + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_market_kline_more_exsymbol(exsymbol_id_lst, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))

    const1 = 'select close, amount, count, high, low, open, period, tickid, volume, exsymbolid from market_kline ' \
             'where exsymbolid in ('
    if exsymbol_id_lst is not None:
        for i in range(len(exsymbol_id_lst)):
            const1 = const1 + '\'' + str(exsymbol_id_lst[i]) + '\'' + ','
    const1 = const1[:-1] + ') and period like ' + '\'' + period + '\'' + ' and tickid between ' \
        + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(const1).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline(exchange, symbol, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, high, low, open, period, tickid, volume, symbol  from kline where exchange ' \
           '= ' + '\'' + exchange + '\'' + ' and period like ' + '\'' + period + '\'' + ' and symbol like ' + '\'' + \
           symbol + '\'' + ' and tickid between ' + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + \
           ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline_period(exchange, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    stmt = 'select close, amount, high, low, open, period, tickid, volume from kline where exchange ' \
           '= ' + '\'' + exchange + '\'' + ' and period like ' + '\'' + period + '\'' + \
           ' and tickid between ' + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(stmt).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_kline_more_exsymbol(exchange, symbol_lst, period, s_time, e_time):
    s_time = str(int(time.mktime(time.strptime(s_time, '%Y-%m-%d %H:%M:%S'))))
    e_time = str(int(time.mktime(time.strptime(e_time, '%Y-%m-%d %H:%M:%S'))))
    const1 = 'select close, amount, high, low, open, period, tickid, volume, symbol from kline where exchange ' \
             '= ' + '\'' + exchange + '\'' + ' and symbol in ('
    if symbol_lst is not None:
        for i in range(len(symbol_lst)):
            const1 = const1 + '\'' + symbol_lst[i] + '\'' + ','
    const1 = const1[:-1] + ') and period like ' + '\'' + period + '\'' + ' and tickid between ' \
        + '\'' + s_time + '\'' + ' and ' + '\'' + e_time + '\'' + ' order by tickid'
    csd_hq = MysqlDBConnector('production')
    data = csd_hq.get_data_from_query(const1).assign(
        date_time=lambda df: df.tickid.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))))
    return data


def get_winR_odds(ret_lst):
    win_lst = [i for i in ret_lst if i > 0]
    loss_lst = [i for i in ret_lst if i < 0]
    win_R = 0
    odds = 1
    ave = 0
    mid_ret = 0
    if len(win_lst) + len(loss_lst) > 0:
        win_R = len(win_lst) / (len(win_lst) + len(loss_lst))
        ave = (sum(win_lst) + sum(loss_lst)) / (len(win_lst) + len(loss_lst))
        odds = 10
        if len(win_lst) == 0:
            win_lst = [0]
        if len(loss_lst) > 0:
            odds = - np.mean(win_lst) / np.mean(loss_lst)
        win_lst.extend(loss_lst)
        mid_ret = np.percentile(win_lst, 50)
    return win_R, odds, ave, mid_ret


def get_winR_max(ret_lst):
    win_lst_3 = [i for i in ret_lst if i > 0.03]
    loss_lst_3 = [i for i in ret_lst if i < 0.03]
    win_lst_5 = [i for i in ret_lst if i > 0.05]
    loss_lst_5 = [i for i in ret_lst if i < 0.05]
    win_R_3 = 0
    win_R_5 = 0
    ave_max = 0
    if len(win_lst_3) + len(loss_lst_3) > 0:
        win_R_3 = len(win_lst_3) / (len(win_lst_3) + len(loss_lst_3))
        ave_max = (sum(win_lst_3) + sum(loss_lst_3)) / (len(win_lst_3) + len(loss_lst_3))
    if len(win_lst_5) + len(loss_lst_5) > 0:
        win_R_5 = len(win_lst_5) / (len(win_lst_5) + len(loss_lst_5))

    return win_R_3, win_R_5, ave_max


def maxRetrace(lst, n):
    '''
    :param list:netlist
    :param n:每交易日对应周期数
    :return: 最大历史回撤
    '''
    Max = 0
    new_lst = copy.deepcopy(lst)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]

    for i in range(len(new_lst)):
        if 1 - new_lst[i] / max(new_lst[:i + 1]) > Max:
            Max = 1 - new_lst[i] / max(new_lst[:i + 1])
    return Max


def annROR(netlist, n):
    '''
    :param netlist:净值曲线
    :param n:每交易日对应周期数
    :return: 年化收益
    '''
    return math.pow(netlist[-1] / netlist[0], 252 * n / len(netlist)) - 1


def daysharpRatio(netlist):
    row = []
    for i in range(1, len(netlist)):
        row.append(math.log(netlist[i] / netlist[i - 1]))
    return np.mean(row) / np.std(row)


def yearsharpRatio(netlist, n):
    '''
    :param netlist:
    :param n: 每交易日对应周期数
    :return:
    '''
    row = []
    new_lst = copy.deepcopy(netlist)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), n)]
    for i in range(1, len(new_lst)):
        row.append(math.log(new_lst[i] / new_lst[i - 1]))
    return np.mean(row) / np.std(row) * math.pow(252, 0.5)


def sharp_maxretrace_ann(netlist):
    row = []
    Max = 0
    new_lst = copy.deepcopy(netlist)
    new_lst = [new_lst[i] for i in range(0, len(new_lst), 1440)]
    for i in range(1, len(new_lst)):
        row.append(math.log(new_lst[i] / new_lst[i - 1]))
        if 1 - new_lst[i] / max(new_lst[:i + 1]) > Max:
            Max = 1 - new_lst[i] / max(new_lst[:i + 1])
    return np.mean(row) / np.std(row) * math.pow(252, 0.5), Max, math.pow(netlist[-1] / netlist[0], 252 * 1440 / len(netlist)) - 1


def get_min_n_from_period(period):
    if (period == '5m') | (period == '5min'):
        n = 5
    if (period == '15m') | (period == '15min'):
        n = 15
    if (period == '30m') | (period == '30min'):
        n = 30
    if (period == '60min') | (period == '1hour') | (period == '1h'):
        n = 60
    if (period == '240m') | (period == '4hour') | (period == '4h'):
        n = 240
    if (period == '1day') | (period == '1d'):
        n = 1440
    return n


def weight_Df(signal_df, pre_df, weight, loss_stop, win_stop, fee):
    signal_slect = copy.deepcopy(signal_df)
    pre_data = copy.deepcopy(pre_df)
    date_lst = copy.deepcopy(signal_slect.tickid.tolist())
    pre_data = pre_data.set_index(['tickid', 'symbol'])
    print(pre_data)
    signal_slect_ = signal_slect.set_index(['tickid'])
    lst = []
    row_dict = {}
    row_dict['tickid'] = date_lst[0]
    row_dict['weight'] = 0
    row_dict['symbol'] = []
    row_dict['return'] = 0
    lst.append(row_dict)
    sig_state_lst = []
    for idx, row_ in signal_slect_.iterrows():
        today = idx
        code_lst = copy.deepcopy(row_.signal)
        date_left = [int(i) for i in date_lst if i > today]
        if (len(date_left) >= 1) & (len(code_lst) > 0):
            for code in code_lst:
                # print(code)
                if date_left[0] in pd.DataFrame(lst).set_index(['tickid']).index:
                    try:
                        if code == pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol']:
                            # print('true1:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'])
                            continue
                    except Exception as e:
                        print(e)
                    try:
                        if code in pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist():
                            # print('true2:', pd.DataFrame(lst).set_index(['tickid']).loc[date_left[0]]['symbol'].tolist())
                            continue
                    except Exception as e:
                        print(e)
                code_hq_today = pre_data.loc[today, code]
                if code_hq_today['low1'] <= -loss_stop:
                    today_return = (1 - loss_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 - loss_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code_hq_today['high1'] >= win_stop:
                    today_return = (1 + win_stop) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (1 + win_stop) * (1 - fee) * (1 - fee)-1
                    continue
                elif code not in signal_slect_.loc[date_left[0]]['signal']:
                    today_return = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee) * weight
                    row_dict = {}
                    row_dict['tickid'] = date_left[0]
                    row_dict['weight'] = weight
                    row_dict['symbol'] = code
                    row_dict['return'] = today_return
                    lst.append(row_dict)
                    ret = (code_hq_today['pre1'] + 1) * (1 - fee) * (1 - fee)-1
                    continue
                today_return = (code_hq_today['pre1'] + 1) * weight * (1-fee)
                row_dict = {}
                row_dict['tickid'] = date_left[0]
                row_dict['weight'] = weight
                row_dict['symbol'] = code
                row_dict['return'] = today_return
                lst.append(row_dict)
                if len(date_left) > 1:
                    m = 1
                    for tickid in date_left[1:]:

                        m = m + 1
                        m2 = m - 1
                        prenextname = 'pre' + str(m) + 'next'
                        prename = 'pre' + str(m2)
                        lowprename = 'low' + str(m)
                        highprename = 'high' + str(m)
                        prelst = []
                        prelst.append(0.0)
                        for n in range(1, m):
                            premaxname = 'pre' + str(n)
                            prelst.append(code_hq_today[premaxname])
                        if code_hq_today[highprename] >= win_stop:
                            today_return = (1 + win_stop) / (1+code_hq_today[prename]) * (1 - fee) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (win_stop + 1) * (1 - fee) * (1 - fee)-1
                            break
                        elif (code_hq_today[lowprename] - max(prelst))/(1+max(prelst)) < -loss_stop:
                            today_return = (1+max(prelst)) * (1-loss_stop)/(1+code_hq_today[prename]) * weight * (1-fee)
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = ((max(prelst) + 1) * (1-loss_stop)) * (1 - fee) * (1 - fee)-1
                            break
                        elif code not in signal_slect_.loc[tickid]['signal']:
                            today_return = (code_hq_today[prenextname] + 1) * (1 - fee) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                            break
                        else:
                            today_return = (code_hq_today[prenextname] + 1) * weight
                            row_dict = {}
                            row_dict['tickid'] = tickid
                            row_dict['weight'] = weight
                            row_dict['symbol'] = code
                            row_dict['return'] = today_return
                            lst.append(row_dict)
                            ret = (code_hq_today['pre' + str(m)] + 1) * (1 - fee) * (1 - fee)-1
                    sig_state_row = []
                    sig_state_row.append(today)
                    sig_state_row.append(code)
                    sig_state_row.append(ret)
                    sig_state_row.append(m)
                    sig_state_lst.append(sig_state_row)
    weight_df = pd.DataFrame(lst)\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    sig_state_df = pd.DataFrame(sig_state_lst, columns=['tickid', 'symbol', 'ret', 'holddays'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return weight_df, sig_state_df


def net_Df(weight_df, signal_slect):
    weight_df = weight_df.assign(tickid=lambda df: df.tickid.apply(lambda x: int(x)))
    weight_df = weight_df.set_index(['tickid'])
    date_lst = [int(i) for i in copy.deepcopy(signal_slect.tickid.tolist())]
    net = 1
    lst = []
    for tickid in date_lst:
        if tickid in weight_df.index:
            change = weight_df.loc[tickid]['return'].sum() + (1-weight_df.loc[tickid]['weight'].sum())

            pos = len(weight_df.loc[tickid]['symbol'])
            if pos > 3:
                pos = 1
        else:
            change = 1.0
            pos = 0
        row_ = []
        row_.append(tickid)
        row_.append(change)
        net = net * change
        row_.append(net)
        row_.append(pos)
        lst.append(row_)
    net_df = pd.DataFrame(lst, columns=['tickid', 'change', 'net', 'pos'])\
        .assign(date_time=lambda df: df.tickid.apply(lambda x: datetime.datetime.fromtimestamp(x)))
    return net_df


def get_predata(hqdata, maxholdday):
    df_lst = []
    for name, group in hqdata.groupby('symbol'):
        t0 = time.time()
        if group.shape[0] < 20:
            print("Stock {stk} has only {cnt} data and skip it.".format(stk=name, cnt=group.shape[0]))
            continue
        group = group.set_index('symbol')
        _df_lst = [group, ]
        df_shift_1 = group.loc[:, ['open', ]].shift(-1)
        for p in range(1, maxholdday + 1, 1):
            df_shift = group.loc[:, ['close', 'low', 'high']].shift(-p).rename(
                columns={"close": "c_p", 'low': "l_p", "high": "h_p"})
            df_shift_ = group.loc[:, ['close', ]].shift(-p + 1).rename(columns={"close": "c_p_"})
            _df = pd.concat([df_shift_1, df_shift, df_shift_], axis=1) \
                .assign(pre=lambda df: (df.c_p / df.open) - 1) \
                .assign(pre_next=lambda df: (df.c_p / df.c_p_ - 1)) \
                .assign(low=lambda df: (df.l_p / df.open - 1)) \
                .assign(high=lambda df: (df.h_p / df.open - 1)) \
                .rename(
                columns={'pre': 'pre{p}'.format(p=p), 'pre_next': 'pre{p}next'.format(p=p), 'low': 'low{p}'.format(p=p),
                         'high': 'high{p}'.format(p=p)}) \
                .drop(['c_p', 'l_p', 'h_p', 'open', 'c_p_'], axis=1)
            _df_lst.append(_df)
        df = pd.concat(_df_lst, axis=1)
        df_lst.append(df)
        print(time.time() - t0)
    dealdataall = pd.concat(df_lst)
    dealdataall = dealdataall.reset_index(drop=False)
    return dealdataall


def get_hqdata_all(symble_lst, period):
    hqdata_all_lst = []
    for symbol in symble_lst:
        data = pd.read_csv('data/' + symbol + period + '.csv',header=None)
        print(data)
        data.columns=['tickid', 'date_time', 'open', 'high', 'low', 'close', 'volume']
        data=data.loc[
             :, ['tickid', 'close', 'high', 'low', 'open']].assign(symbol=symbol)
        hqdata_all_lst.append(data)
    return pd.concat(hqdata_all_lst).reset_index(drop=True)


if __name__ == '__main__':
    os.chdir(r'/Users/zhangfang/PycharmProjects')

    # data = get_kline('BIAN', 'btcusdt', '1m', '2018-08-17 00:00:00', '2018-10-01 00:00:00')
    # print(data)
    # data.to_csv('data/btcusdt_1m.csv')
    symbol_lst = ['ethbtc', 'eosbtc', 'etcbtc', 'iotabtc', 'iostbtc', 'ltcbtc', 'neobtc', 'trxbtc', 'xrpbtc', 'xlmbtc',
                'adabtc', 'ontbtc', 'bnbbtc', 'bchabcbtc', 'bchsvbtc', "mdabtc", "stratbtc", "xmrbtc", "dashbtc",
                "xembtc", "zecbtc", "wavesbtc", "btgbtc", "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc"]
    # symbol_lst = ['btcusdt', 'ethusdt', 'eosusdt', 'etcusdt', 'iotausdt', 'iostusdt', 'ltcusdt', 'neousdt', 'trxusdt',
    #             'xrpusdt', 'xlmusdt', 'adausdt', 'ontusdt', 'bnbusdt', 'bchusdt', 'bchsvusdt', "xmrusdt", "dashusdt",
    #             "wavesusdt", "vetusdt", "qtumusdt", "zrxusdt"]
    for period in ['1440']:
        hqdata = get_hqdata_all(symbol_lst, period)
        print(hqdata)
        hqdata.to_csv('data/coinbase_btc_28_' + period + '.csv')
    for period in ['1440']:
        hqdata = pd.read_csv('data/coinbase_btc_28_' + period + '.csv').loc[
             :, ['tickid', 'close', 'high', 'low', 'open', 'symbol']]
        pre_data = get_predata(hqdata, 150)
        print(pre_data)
        pre_data.to_hdf('data/coinbase_btc_28_predata' + period + 'm.h5', 'all')


