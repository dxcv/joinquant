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


def rpt(dfin, N):
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
            <h1 align='center'>建议 </h1>\
        </div>\
        <hr>\
        <div class='content'>\
            <!--正文内容-->\
            <h2>建议：</h2>\
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


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    today = datetime.date.today()
    fold = 'E:/fof/cgo/'
    fund = ['399006.XSHE', '000300.XSHG', '000905.XSHG']
    # fund = ['000016.XSHG']
    indus_name_lst = ['cyb', 'hs300', 'zz500']
    N = 100
    num = 0
    bars = 252
    calen = get_trade_days(count=bars)
    calen = list(calen)
    if today in calen:
        next_tradeday = get_trade_days(start_date=today+datetime.timedelta(days=1), end_date='2030-01-01')[0]
        calen = list(calen)
        if datetime.datetime.now().hour > 18:
            calen.append(next_tradeday)
        EndDate = calen[-1]
        today = EndDate
        StartDate = calen[0]
        listDATE = calen[-125]
        listday_info = get_all_securities(types=['stock'], date=today).query(
            "start_date<'{date}'".format(date=listDATE))
        all_stocks_allowed = listday_info.index.tolist()
        print(all_stocks_allowed)
        cgo_dict = {}
        for indus_name in fund:
            SZ50_stocks_list = index_stocks(indus_name)
            SZ50_stocks_list = normalize_code(SZ50_stocks_list)
            SZ50_stocks_list = [i for i in SZ50_stocks_list if i in all_stocks_allowed]
            ratio_list = list()
            values = {'cumprod1': 1}
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

            # =============================================================================
            # 计算逻辑
            # =============================================================================
            df_50_1 = df_50_1[['stockcode', 'tradedate', 'turn_radio', 'close', 'volume', 'money']]
            df_50_1.columns = ['STOCKCODE', 'ENDDATE', 'TURNOVER_DAY', 'TCLOSE', 'TVOLUME', 'TVALUE']
            df_50_1 = df_50_1.reset_index(drop=True)\
                .assign(avgprice=lambda df: df.TVALUE/df.TVOLUME)\
                .assign(TURNOVER_DAY=lambda df: df.TURNOVER_DAY/100)\
                .assign(comturnover=lambda df: 1-df.TURNOVER_DAY.shift(-1))

            # df_50_1.to_csv(fold + 'stock_hq_' + indus_name[:6] + '.csv', encoding='gbk')
            # df_50_1 = pd.read_csv(fold + 'stock_hq_' + indus_name[:6] + '.csv', encoding='gbk', index_col=0)

            date = today

            # 计算上证50的CGO
            num = 0
            out = []

            for name, group in df_50_1.groupby('STOCKCODE'):

                temp = group.reset_index(drop=True).assign(rpt=lambda df: [
                    rpt(df.iloc[get_index(i):i, :], N) for i in
                    range(1, len(group) + 1)]).assign(STOCKCODE=name)
                out.append(temp)
                num = num + 1
                if num % 10 == 0:
                    print(num)

            out = pd.concat(out)\
                .assign(CGO=lambda df: (df.TCLOSE - df.rpt) / df.rpt)[['ENDDATE', 'CGO']]\
                .assign(trade_date=lambda df: df.ENDDATE.apply(lambda x: str(x)[:10]))[['trade_date', 'CGO']]
            # out.to_csv(fold + 'stock_cgo_' + indus_name[:6] + '.csv', encoding='gbk')
            cgo_dict[indus_name[:6]] = out

            ratio = out.query("ENDDATE=='{date1}'".format(date1=max(out.ENDDATE))).dropna()
            ratio_list.append(len(ratio.query("CGO>0")) / len(ratio))

            f = lambda s: s.quantile(0.3)
            # f2 = lambda s: s.quantile(0.5)
            cgo = out.groupby('ENDDATE') \
                .CGO \
                .apply(f) \
                .reset_index()

            res = cgo
            res = res.rename(index=str, columns={'CGO': 'CGO_50'})

            res = res.assign(macgo_50_20=lambda df: tb.MA(df.CGO_50.values, 15)) \
                .assign(macgo_50_10=lambda df: tb.MA(df.CGO_50.values, 5))\
                .assign(position=lambda df: df.macgo_50_10 > df.macgo_50_20)\
                .assign(position=lambda df: df.position.apply(lambda x: int(x)))

            # =============================================================================
            # 邮件发送
            # =============================================================================
            #    print('完成计算')
            # 整理数据
            res.dropna(inplace=True)
            res.to_csv(fold + 'res_' + indus_name[:6] + '.csv', encoding='gbk')
            # res = pd.read_csv(fold + 'res_' + indus_name + '.csv', encoding='gbk', index_col=0)

            # res = res.assign(sign2=[sign2(row) for idx, row in res.iterrows()])
            # row_ = res.query("ENDDATE=='{date1}'".format(date1=max(out.ENDDATE)))

        #     position_last = res.position.tolist()[-1] * ratio_list[0]
        #
        #     res = res[['ENDDATE', 'sign2']]
        #     now = datetime.datetime.now()
        #     now = now.strftime('%Y-%m-%d')
        #     res['UpdateTime'] = now
        #     res = res.sort_values(by='ENDDATE', ascending=False)
        #     res = res.assign(sign_chinese=lambda df: df.sign2.apply(f_s1))
        #     # 发送邮件
        #     # =============================================================================
        #     # 将信息发送邮件
        #     # =============================================================================
        #     # 设置smtplib所需的参数
        #     # 下面的发件人，收件人是用于邮件传输的。
        #     smtpserver = 'smtp.163.com'
        #     username = 'xiahutao@163.com'
        #     password = 'yo425408'
        #     sender = 'xiahutao@163.com'
        #     # receiver='XXX@126.com'
        #     # 收件人为多个收件人
        #     # receiver=['542362275@qq.com','43521385@qq.com','519518384@qq.com','xiaoliang.jia@jrj.com.cn','rdfce6@126.com','zxdokok@sina.com','654497127@qq.com','1518139212@qq.com','18244689637@163.com','kakaiphoimo@163.com','419393571@qq.com','554108967@qq.com','yanghui_hu@sina.com']
        #     receiver = ['zxdokok@sina.com','43521385@qq.com','542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
        #     # receiver = [' xiahutao@163.com', '519518384@qq.com']
        #
        #     res.index = range(len(res))
        #     # date=res.iloc[0,:]['ENDDATE']
        #
        #     today = today.strftime('%Y-%m-%d')
        #     date = today
        #     context = res.iloc[0, :]['sign_chinese']
        #     # context1=res.iloc[0,:]['sign_position']
        #     # context1 = '大盘股建议仓位:' + str(int(position_last[0])) + '%' + ' ' + '蓝筹股建议仓位：' + str(
        #     #     int(position_last[2])) + '%' + ' ' + '小盘股建议仓位：' + str(int(position_last[1])) + '%' + ' ' + '整体仓位建议：不超过' + str(
        #     #     int(sum(position_last))) + '%'
        #     context1 = '大:' + str(int(position_last[0])) + '%' + '蓝：' + str(
        #         int(position_last[2])) + '%' + '小：' + str(
        #         int(position_last[1])) + '%' + '整体：不超过' + str(
        #         int(sum(position_last))) + '%'
        #
        #     res_n = res.copy().loc[:, ['ENDDATE', 'sign_chinese']]
        #     res_n.ENDDATE = res_n.ENDDATE.shift(1)
        #     res_n.ENDDATE = res_n.ENDDATE.fillna(today)
        #     res_n.columns = ['日期', 'resualt']
        #     print(res_n)
        #
        #     # subject = date + ' ' + context1
        #     subject = date + '今日适合:' + context + ',' + '建议:' + context1
        #     # 通过Header对象编码的文本，包含utf-8编码信息和Base64编码信息。以下中文名测试ok
        #     # subject = '中文标题'
        #     # subject=Header(subject, 'utf-8').encode()
        #
        #     # 构造邮件对象MIMEMultipart对象
        #     # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
        #     msg = MIMEMultipart('mixed')
        #     msg['Subject'] = subject
        #     msg['From'] = 'xiahutao@163.com <xiahutao@163.com>'
        #     # msg['To'] = 'XXX@126.com'
        #     # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
        #     msg['To'] = ";".join(receiver)
        #     # msg['Date']='2012-3-16'
        #
        #     # 构造文字内容
        #     text = '今日适合:' + context + ',' + '建议:' + context1
        #     text_plain = MIMEText(text, 'plain', 'utf-8')
        #     msg.attach(text_plain)
        #     html_msg = get_html_msg(res_n)
        #     content_html = MIMEText(html_msg, "html", "utf-8")
        #     msg.attach(content_html)
        #
        #     # 发送邮件
        #     smtp = smtplib.SMTP()
        #     smtp.connect('smtp.163.com')
        #     smtp.login(username, password)
        #     smtp.sendmail(sender, receiver, msg.as_string())
        #     smtp.quit()
        # #    print('完成邮件发送')
