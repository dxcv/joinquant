# -*- coding: utf-8 -*-
# @Time    : 2020/4/10 16:20
# @Author  : zhangfang

from __future__ import division
from notify.notifyapi import *
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

def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True,
                     fq='post',
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


def positon(row):
    out = list()
    if (row['macgo_50_10'].values[0] > row['macgo_50_20'].values[0]):
        out.append(1)
    else:
        out.append(0)

    return out


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
        .rename(columns={'index': 'trade_date'}) \
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
    # receiver = ['xiahutao@163.com', '3467518502@qq.com', '542362275@qq.com']
    receiver = ['xiahutao@163.com', '542362275@qq.com']
    today = datetime.date.today()
    fold = 'E:/fof/cgo/'
    calen = get_trade_days(count=10)

    if today in calen:
        code_df = pd.read_csv('G:/lfp/price_line.csv', encoding='gbk')
        code_df['stock_code'] = code_df['stock_code'].apply(lambda x: normalize_code(x))
        code_df['price'] = code_df['price'].apply(
            lambda x: float(str(x)[6:]))
        lst = []
        for idx, row in code_df.iterrows():
            price_now = stock_price(row.stock_code, '1d', today, today).low.tolist()[0]
            lst.append([row.stock_code, row.price, price_now])
        ret = pd.DataFrame(lst, columns=['code', 'line', 'price_now']) \
            .assign(warnning=lambda df: df.price_now <= df.line) \
            .sort_values(['warnning'], ascending=False).assign(code=lambda df: df.code.apply(lambda x: x[:6]))[
            ['code', 'line', 'price_now', 'warnning']]
        print(ret)
        context1 = '---WARNNING'
        res_n = ret.copy().reset_index(drop=True)
        res_n = res_n[res_n['warnning'] == True]
        subject = str(today)[:10] + context1
        send_email(ret.copy().reset_index(drop=True), subject, From, receiver)
        if len(res_n) > 0:
            errcode, errmsg = send_yeecall("8615624980748", subject, '触发价格底线股票: ' + '_'.join(res_n.code.tolist()))
            print(errcode, errmsg)