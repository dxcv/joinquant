# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:21:02 2018

@author: lion95
"""

from __future__ import division
import pandas as pd
import os
import numpy as np
from jqdatasdk import *

import copy
auth('18610039264', 'zg19491001')
import datetime
import talib as tb
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from smtplib import SMTP_SSL


# =============================================================================
# 函数
# =============================================================================

def values_data_cgo(stockcode, count, eday):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe 市值表 字段为 code :股票代码  day:日期  capitalization:总股本（万股）
    circulating_cap ：流通股本（万股） market_cap：总市值（亿） circulating_market_cap：流通市值（亿）
    turnover_ratio：换手率 pe_ratio：市盈率 TTM pe_ratio_lyr：静态市盈率  pb_ratio：市净率
    ps_ratio：市销率  pcf_ratio：市现率
    """
    q = query(valuation.code,
              valuation.turnover_ratio,
              ).filter(valuation.code == stockcode)

    df = get_fundamentals_continuously(q, count=count, end_date=eday, panel=False)[['day', 'code', 'turnover_ratio']]

    return df


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='post',
                     count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


# 获取当前各大指数的成分股
def index_stocks(_index):
    """
    输入 指数编码：000016.XSHG	上证50；000300.XSHG	沪深300；399005.XSHE	中小板指
                 399006.XSHE	创业板指；000905.XSHG	中证500
    返回 成分股代码列表
    输出格式 list
    """
    return get_index_stocks(_index)


def rpt(dfin, N, values):
    if len(dfin) < N:
        return dfin.TCLOSE.mean()
    else:
        s0 = dfin.copy()
        s0 = s0.sort_values(by='ENDDATE', ascending=False)
        s0.index = range(len(s0))
        s0 = s0.assign(cumprod=lambda df: df.comturnover.cumprod()) \
            .assign(cumprod1=lambda df: df['cumprod'].shift(1)) \
            .fillna(value=values) \
            .assign(turnoverUP=lambda df: df.TURNOVER_DAY * df.cumprod1) \
            .assign(turnovergui=lambda df: df.turnoverUP / df.turnoverUP.sum()) \
            .assign(rp=lambda df: df.avgprice * df.turnovergui)
        return s0.rp.sum()


def get_index(i):
    if i < N:
        return 0
    else:
        return i - N


def sign(row):
    if (row['whobig'] == True) & (row['cold'] == False):
        return 1
    elif (row['whobig'] == False) & (row['cold'] == False):
        return -1
    return 0


def distribute(SE):
    low = SE.quantile(0.05)
    high = SE.quantile(0.95)
    res = SE[(SE > low) & (SE < high)]
    return res


def sign2(row):
    if ((row['macgo_500_10'] > row['macgo_500_20']) & (row['macgo_50_10'] > row['macgo_50_20']) & (
            row['macgo_300_10'] > row['macgo_300_20'])):
        return 1
    elif ((row['macgo_500_10'] < row['macgo_500_20']) & (row['macgo_50_10'] > row['macgo_50_20']) & (
            row['macgo_300_10'] > row['macgo_300_20'])):
        return 2
    elif ((row['macgo_500_10'] > row['macgo_500_20']) & (row['macgo_50_10'] < row['macgo_50_20']) & (
            row['macgo_300_10'] > row['macgo_300_20'])):
        return 3
    elif ((row['macgo_500_10'] > row['macgo_500_20']) & (row['macgo_50_10'] > row['macgo_50_20']) & (
            row['macgo_300_10'] < row['macgo_300_20'])):
        return 4
    elif ((row['macgo_500_10'] > row['macgo_500_20']) & (row['macgo_50_10'] < row['macgo_50_20']) & (
            row['macgo_300_10'] < row['macgo_300_20'])):
        return 5
    elif ((row['macgo_500_10'] < row['macgo_500_20']) & (row['macgo_50_10'] > row['macgo_50_20']) & (
            row['macgo_300_10'] < row['macgo_300_20'])):
        return 6
    elif ((row['macgo_500_10'] < row['macgo_500_20']) & (row['macgo_50_10'] < row['macgo_50_20']) & (
            row['macgo_300_10'] > row['macgo_300_20'])):
        return 7
    else:
        return 0


def positon(row):
    out = list()
    if (row['macgo_50_10'].values[0] > row['macgo_50_20'].values[0]):
        out.append(1)
    else:
        out.append(0)

    return out


f_s1 = lambda \
        s: '大，蓝，小' if s == 1 else '大，蓝' if s == 2 else '蓝，小' if s == 3 else '大，小' if s == 4 else '小' if s == 5 else '大' if s == 6 else '蓝' if s == 7 else '空'


def get_html_msg(data):
    """
    1. 构造html信息
    """
    df = data.copy()
    #    df=df.iloc[-7:,:]
    #    df=df.sort_values(by='日期',ascending=False)
    #    df.index=range(len(df))
    #    df['缩略图'] = '<img data-src="' + df['缩略图'] + '">'
    df_html = df.to_html(escape=False)

    head = \
        "<head> \
            <meta charset='utf-8'> \
            <STYLE TYPE='text/css' MEDIA=screen> \
                table.dataframe { \
                    border-collapse: collapse;\
                    border: 2px solid \
                    /*居中显示整个表格*/ \
                    margin: auto; \
                } \
                table.dataframe thead { \
                    border: 2px solid #91c6e1;\
                    background: #f1f1f1;\
                    padding: 10px 10px 10px 10px;\
                    color: #333333;\
                }\
                table.dataframe tbody {\
                    border: 2px solid #91c6e1;\
                    padding: 10px 10px 10px 10px;\
                }\
                table.dataframe tr { \
                } \
                table.dataframe th { \
                    vertical-align: top;\
                    font-size: 14px;\
                    padding: 10px 10px 10px 10px;\
                    color: #105de3;\
                    font-family: arial;\
                    text-align: center;\
                }\
                table.dataframe td { \
                    text-align: center;\
                    padding: 10px 10px 10px 10px;\
                }\
                body {\
                    font-family: 宋体;\
                }\
                h1 { \
                    color: #5db446\
                }\
                div.header h2 {\
                    color: #0002e3;\
                    font-family: 黑体;\
                }\
                div.content h2 {\
                    text-align: center;\
                    font-size: 28px;\
                    text-shadow: 2px 2px 1px #de4040;\
                    color: #fff;\
                    font-weight: bold;\
                    background-color: #008eb7;\
                    line-height: 1.5;\
                    margin: 20px 0;\
                    box-shadow: 10px 10px 5px #888888;\
                    border-radius: 5px;\
                }\
                h3 {\
                    font-size: 22px;\
                    background-color: rgba(0, 2, 227, 0.71);\
                    text-shadow: 2px 2px 1px #de4040;\
                    color: rgba(239, 241, 234, 0.99);\
                    line-height: 1.5;\
                }\
                h4 {\
                    color: #e10092;\
                    font-family: 楷体;\
                    font-size: 20px;\
                    text-align: center;\
                }\
                td img {\
                    /*width: 60px;*/\
                    max-width: 300px;\
                    max-height: 300px;\
                }\
            </STYLE>\
        </head>\
        "
    # 构造模板的附件（100）
    body = "<body>\
        <div align='center' class='header'> \
            <!--标题部分的信息-->\
            <h1 align='center'>  </h1>\
        </div>\
        <hr>\
        <div class='content'>\
            <!--正文内容-->\
            <h2> ：</h2>\
            <div>\
                <h4></h4>\
                {df_html}\
            </div>\
            <hr>\
            <p style='text-align: center'>\
                —— 本次报告完 ——\
            </p>\
        </div>\
        </body>\
        ".format(df_html=df_html)
    html_msg = "<html>" + head + body + "</html>"
    # 这里是将HTML文件输出，作为测试的时候，查看格式用的，正式脚本中可以注释掉
    fout = open('t4.html', 'w', encoding='UTF-8', newline='')
    fout.write(html_msg)
    return html_msg


def stock_code_lst(sec):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['wind_index'].read(sec + '_component')
    temp = temp[temp['date'] == temp['date'].max()]
    return temp.code.tolist()


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour > 18:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    listDATE = calen[-125]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, listDATE, StartDate, str(hq_last_date)[:10]


def get_cgo_dict(index_code_lst, name_lst):
    cgo_dict = {}
    values = {'cumprod1': 1}
    for i in range(len(index_code_lst)):
        code = index_code_lst[i]
        index_name = name_lst[i]
        SZ50_stocks_list = index_stocks(code)
        SZ50_stocks_list = normalize_code(SZ50_stocks_list)
        SZ50_stocks_list = [i for i in SZ50_stocks_list if i in all_stocks_allowed]

        #  合成50成分股数据
        print('==================================================================================')
        df_50_turnover = []
        for i in SZ50_stocks_list:
            temp = values_data_cgo(i, bars, EndDate)
            df_50_turnover.append(temp)
        df_50_turnover = pd.concat(df_50_turnover)
        df_50_turnover.columns = ['tradedate', 'stockcode', 'turn_radio']
        print(pd.DataFrame(df_50_turnover))
        df_50_turnover['tradedate'] = df_50_turnover['tradedate'].apply(lambda x: str(x)[:10])
        # print(df_50_turnover)
        # 行情数据
        df_50_price = []
        for i in SZ50_stocks_list:
            temp = stock_price_cgo(i, StartDate, EndDate)
            df_50_price.append(temp)
        df_50_price = pd.concat(df_50_price).assign(tradedate=lambda df: df.tradedate.apply(lambda x: str(x)[:10]))
        print(df_50_price)
        df_50_1 = df_50_turnover.merge(df_50_price, on=['tradedate', 'stockcode'])
        print('==================================================================================')

        df_50_1 = df_50_1[['stockcode', 'tradedate', 'turn_radio', 'close', 'volume', 'money']]
        df_50_1.columns = ['STOCKCODE', 'ENDDATE', 'TURNOVER_DAY', 'TCLOSE', 'TVOLUME', 'TVALUE']
        df_50_1 = df_50_1.reset_index(drop=True) \
            .assign(avgprice=lambda df: df.TVALUE / df.TVOLUME) \
            .assign(TURNOVER_DAY=lambda df: df.TURNOVER_DAY / 100) \
            .assign(comturnover=lambda df: 1 - df.TURNOVER_DAY.shift(-1))
        # 计算上证50的CGO
        num = 0
        out = []
        for name, group in df_50_1.groupby('STOCKCODE'):
            temp = group.reset_index(drop=True).assign(rpt=lambda df: [
                rpt(df.iloc[get_index(i):i, :], N, values) for i in
                range(1, len(group) + 1)]).assign(STOCKCODE=name)
            out.append(temp)
            num = num + 1
            if num % 10 == 0:
                print(num)
        out = pd.concat(out) \
            .assign(CGO=lambda df: (df.TCLOSE - df.rpt) / df.rpt)[['ENDDATE', 'CGO']] \
            .assign(trade_date=lambda df: df.ENDDATE.apply(lambda x: str(x)[:10]))[['trade_date', 'CGO']]
        cgo_dict[index_name] = out
    return cgo_dict


def get_ymjh_pos_df_all(index_code_lst, name_lst, index_hq_dic, para_dict, hq_last_date):
    pos_df_all = pd.DataFrame(columns=['trade_date'])

    for j in range(len(index_code_lst)):
        index_code = index_code_lst[j]
        pos_df = pd.DataFrame(columns=['trade_date'])
        index_name = name_lst[j]
        index_hq = index_hq_dic[index_code]
        index_hq = index_hq[index_hq['trade_date'] <= hq_last_date]

        para_lst = para_dict[index_name]
        for i in range(len(para_lst)):
            para = para_lst[i]
            N1 = para[0]
            N2 = para[1]
            position = index_hq \
                .assign(HH_s=lambda df: tb.MAX(df.high.values, N1)) \
                .assign(LL_s=lambda df: tb.MIN(df.low.values, N1)) \
                .assign(HH_l=lambda df: tb.MAX(df.high.values, N2)) \
                .assign(LL_l=lambda df: tb.MIN(df.low.values, N2)) \
                .assign(ma_s=lambda df: (df.HH_s + df.LL_s) / 2) \
                .assign(ma_l=lambda df: (df.HH_l + df.LL_l) / 2) \
                .assign(ma_s1=lambda df: df.ma_s.shift(1)) \
                .assign(ma_l1=lambda df: df.ma_l.shift(1)) \
                .assign(ave_p=lambda df: (2 * df.close + df.high + df.low) / 4)
            pos = 0
            pos_lst = []
            for idx, _row in position.iterrows():
                condition_l = ((_row.ma_s1 == _row.ma_l1) and (_row.ma_s > _row.ma_l) and (
                        _row.ave_p >= _row.ma_s)) or ((_row.ma_s1 < _row.ma_l1) and (
                        _row.ma_s > _row.ma_l) and (_row.ave_p >= min(_row.ma_s, _row.ma_l)))
                condition_s = (_row.ma_s1 > _row.ma_l1) and (_row.ma_s < _row.ma_l) and (
                        _row.ave_p <= max(_row.ma_s, _row.ma_l))
                if pos == 0:
                    if condition_l:
                        pos = 1
                elif pos == 1:
                    if condition_s:
                        pos = 0
                pos_lst.append(pos)
            position['pos' + str(i)] = pos_lst
            position = position.reset_index(drop=True)[['trade_date', 'pos' + str(i)]]
            pos_df = pos_df.merge(position, on=['trade_date'], how='outer')
        pos_df = pos_df.dropna().sort_values(['trade_date']).set_index(['trade_date'])

        pos_df[index_name + '_ymjh'] = pos_df.sum(axis=1) / len(para_lst)
        pos_df_all = pos_df_all.merge(
            pos_df.reset_index(drop=False)[['trade_date', index_name + '_ymjh']], on=['trade_date'], how='outer')
    return pos_df_all


def get_pos_df_all(index_code_lst, name_lst, cgo_dict, para_dict):
    pos_df_all = pd.DataFrame(columns=['trade_date'])
    ratio_dict = {}
    for j in range(len(index_code_lst)):
        index_name = name_lst[j]
        cgo_ori = cgo_dict[index_name]
        para_lst = para_dict[index_name]
        pos_df = pd.DataFrame(columns=['trade_date'])
        for i in range(len(para_lst)):
            para = para_lst[i]
            pecent = para[0]
            s_period = para[1]
            l_period = para[2]
            f = lambda s: s.quantile(pecent)
            cgo = cgo_ori.groupby('trade_date').CGO.apply(f).reset_index()
            position = cgo.assign(cgo_ma_l=lambda df: tb.MA(df.CGO.values, l_period)) \
                .assign(cgo_ma_s=lambda df: tb.MA(df.CGO.values, s_period)) \
                .assign(position=lambda df: df.cgo_ma_s > df.cgo_ma_l) \
                .assign(position=lambda df: df.position.apply(lambda x: int(x)))[['trade_date', 'position']] \
                .rename(columns={'position': 'position' + str(i)}).dropna()
            pos_df = pos_df.merge(position, on=['trade_date'], how='outer')
        pos_df = pos_df.dropna().sort_values(['trade_date']).set_index(['trade_date'])
        pos_df[index_name + '_cgo'] = pos_df.sum(axis=1) / len(para_lst)
        pos_df_all = pos_df_all.merge(
            pos_df.reset_index(drop=False)[['trade_date', index_name + '_cgo']], on=['trade_date'], how='outer')
        ratio = cgo_ori.query("trade_date=='{date1}'".format(date1=max(cgo_ori.trade_date))).dropna()
        ratio_dict[index_name] = len(ratio.query("CGO>0")) / len(ratio)
    pos_df_all = pos_df_all.dropna().sort_values(['trade_date']).set_index(['trade_date'])
    # pos_df_all.to_csv(fold + 'indus_pos_df_all_.csv', encoding='gbk')
    return pos_df_all.reset_index(drop=False), ratio_dict


def send_email(resualt, subject, From, receiver):

    # 通过Header对象编码的文本，包含utf-8编码信息和Base64编码信息。以下中文名测试ok
    # subject = '中文标题'
    # subject=Header(subject, 'utf-8').encode()

    # 构造邮件对象MIMEMultipart对象
    # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = From
    # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
    msg['To'] = ";".join(receiver)
    # msg['Date']='2012-3-16'

    # 构造文字内容
    text = subject
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)
    html_msg = get_html_msg(resualt)
    content_html = MIMEText(html_msg, "html", "utf-8")
    msg.attach(content_html)

    # 发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.163.com')
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
    print('完成邮件发送')


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period,
                     skip_paused=False, fq='pre', count=None).reset_index() \
        .rename(columns={'index': 'trade_date'})\
        .assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    temp['stock_code'] = sec
    return temp


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    # 下面的发件人，收件人是用于邮件传输的。
    smtpserver = 'smtp.163.com'
    username = 'xiahutao@163.com'
    password = 'GQYJMWRAYELIJGWK'
    sender = 'xiahutao@163.com'
    From = 'xiahutao@163.com <xiahutao@163.com>'
    # 收件人为多个收件人
    # receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
    # receiver = ['xiahutao@163.com', '3467518502@qq.com', '542362275@qq.com']
    receiver = ['xiahutao@163.com']
    today = datetime.date.today()
    fold = 'E:/fof/cgo/'
    index_code_lst = ['399006.XSHE', '000300.XSHG', '000905.XSHG', '000016.XSHG']
    name_lst = ['cyb', 'hs', 'zz', 'sz']
    para_dict = {'cyb': [(0.7, 8, 18), (0.9, 9, 18), (0.85, 11, 30), (0.85, 20, 38), (0.95, 15, 18)],
                 'hs': [(0.95, 9, 22), (0.95, 12, 18), (0.9, 7, 54), (0.9, 14, 38), (0.9, 8, 12)],
                 'zz': [(0.95, 9, 68), (0.65, 15, 38), (0.95, 14, 62), (0.95, 11, 22), (0.9, 13, 20)],
                 'sz': [(0.95, 7, 28), (0.55, 12, 88), (0.95, 16, 22), (0.95, 10, 18), (0.4, 9, 14)]}
    para_ymjh_dict = {'hs': [(17, 70), (7, 26), (6, 46), (5, 48), (9, 76)],
                 'sz': [(22, 66), (7, 28), (6, 62), (8, 62), (10, 82)],
                 'zz': [(5, 38), (21, 32), (8, 52), (20, 30), (6, 34)],
                 'cyb': [(4, 26), (29, 46), (6, 28), (18, 30), (35, 48)]}
    N = 100
    num = 0
    bars = 252
    calen = get_trade_days(count=bars)
    calen = list(calen)
    if today in calen:
        calen, next_tradeday, EndDate, listDATE, StartDate, hq_last_date = get_date(calen, today)
        listday_info = get_all_securities(types=['stock'], date=EndDate).query(
            "start_date<'{date}'".format(date=listDATE))
        all_stocks_allowed = listday_info.index.tolist()
        print(all_stocks_allowed)
        cgo_dict = get_cgo_dict(index_code_lst, name_lst)
        pos_df_all_cgo, ratio_dict = get_pos_df_all(index_code_lst, name_lst, cgo_dict, para_dict)

        print('完成计算')
        res = pos_df_all_cgo
        now = datetime.datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        res['UpdateTime'] = now
        res = res.sort_values(by='trade_date', ascending=False)

        res.index = range(len(res))
        EndDate = EndDate.strftime('%Y-%m-%d')
        date = EndDate
        context1 = ''
        ratio_list = []
        for m in range(len(name_lst)):
            context1 = context1 + name_lst[m] + ':' + str(int(ratio_dict[name_lst[m]] * 100)) + '% '
            ratio_list.append(ratio_dict[name_lst[m]])
        total = int(np.mean(ratio_list) * 100)
        context1 = context1 + '整体：' + str(total) + '%'

        res_n = res.copy()
        res_n.trade_date = res_n.trade_date.shift(1)
        res_n.trade_date = res_n.trade_date.fillna(EndDate)
        # res_n.columns = ['日期', 'resualt']
        print(res_n)
        subject = date + ' CGO: ' + context1
        send_email(res_n, subject, From, receiver)
        index_hq_dic = {}

        for index_code in index_code_lst:
            index_hq = stock_price(index_code, '1d', StartDate, today)
            index_hq_dic[index_code] = index_hq
        pos_df_all_ymjh = get_ymjh_pos_df_all(index_code_lst, name_lst, index_hq_dic, para_ymjh_dict, hq_last_date)
        print(pos_df_all_ymjh)
        pos_df_all = pos_df_all_cgo.merge(pos_df_all_ymjh)
        for index_name in name_lst:
            pos_df_all[index_name] = pos_df_all[index_name + '_cgo'] * pos_df_all[index_name + '_ymjh']
        print(pos_df_all)
        res = pos_df_all_ymjh.sort_values(by='trade_date', ascending=False)
        res.index = range(len(res))
        res_n = res.copy()
        res_n.trade_date = res_n.trade_date.shift(1)
        res_n.trade_date = res_n.trade_date.fillna(EndDate)
        send_email(res_n, date, From, receiver)

        colume_names = copy.deepcopy(name_lst)
        colume_names.append('trade_date')

        res = pos_df_all.sort_values(by='trade_date', ascending=False)[colume_names]
        res = res.dropna().set_index(['trade_date'])
        res['total'] = res.sum(axis=1) / len(index_code_lst)
        res = res.reset_index(drop=False)
        res.index = range(len(res))
        res_n = res.copy()
        res_n.trade_date = res_n.trade_date.shift(1)
        res_n.trade_date = res_n.trade_date.fillna(EndDate)
        send_email(res_n, date, From, receiver)


