#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:16:45 2019
实盘计算单因子选行业并邮件播报_沪深300版
@author: yeecall
"""

from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import mstats
from scipy import stats
import datetime

import os

from factors_gtja import *
import matplotlib.pyplot as plt
import math
from jqdatasdk import *

auth('18610039264', 'zg19491001')

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from smtplib import SMTP_SSL


def stock_price(sec, period, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency=period, fields=None, skip_paused=False, fq='pre',
                     count=None).reset_index() \
        .rename(columns={'index': 'tradedate'})
    temp['stockcode'] = sec
    return temp


# 获取交易日
def tradeday(eday, count):
    """
    输入 开始时间 和 截止时间
    输出 list 交易日 datetime格式
    """
    return get_trade_days(end_date=eday, count=count)


# 行业编码进行提取
def industries_code(_index):
    """
    输入
    sw_l1: 申万一级行业
    sw_l2: 申万二级行业
    sw_l3: 申万三级行业
    jq_l1: 聚宽一级行业
    jq_l2: 聚宽二级行业
    zjw: 证监会行业

    输出
    dataframe 行业代码，行业名称
    """
    return get_industries(_index).reset_index().rename(columns={'index': 'indust_code'})


# 输出行业成分股
def industries_stocks(code):
    """
    输入 行业代码
    输出 list 个股代码
    """
    return get_industry_stocks(code)


def toname(row, dic):
    re_f = list()
    s = row['stk_rec']
    for i in s:
        re_f.append(dic[i])
    return re_f


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
            <h1 align='center'>重点关注行业 </h1>\
        </div>\
        <hr>\
        <div class='content'>\
            <!--正文内容-->\
            <h2>行业表：</h2>\
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


if __name__ == '__main__':

    period = '1d'
    today = datetime.date.today()
    traday = get_trade_days(count=30)[-11:]
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]

    if datetime.datetime.now().hour > 18:
        traday.append(next_tradeday)
    if today in traday:

        traday = [str(i) for i in traday]

        stocks = get_index_stocks('000300.XSHG')
        print(len(stocks))
        factor = 'Alpha.alpha015'

        print('开始')
        print('合成行业个股明细表')
        indust_lst = industries_code('sw_l1')['indust_code'].tolist()
        #    industries_stocks(indust_lst[0])
        indust_stocks = pd.DataFrame(industries_stocks(indust_lst[0]))
        indust_stocks['indust'] = indust_lst[0]
        indust_stocks.columns = ['stockcode', 'indust']
        for i in indust_lst[1:]:
            try:
                print(i)
                indust_stocks_temp = pd.DataFrame(industries_stocks(i))
                indust_stocks_temp['indust'] = i
                indust_stocks_temp.columns = ['stockcode', 'indust']
                indust_stocks = pd.concat([indust_stocks, indust_stocks_temp])
            except:
                print(i + 'is error')

        industry_code_name = industries_code('sw_l1')

        code_to_name = dict(zip(industry_code_name.indust_code.tolist(), industry_code_name.name.tolist()))

        stock_code_name = get_all_securities().reset_index()
        stock_code_name = stock_code_name.iloc[:, :2]
        stk_code_to_name = dict(zip(stock_code_name.iloc[:, 0].tolist(), stock_code_name.iloc[:, 1].tolist()))

        num = 0
        df_stk300 = pd.DataFrame()
        for i in stocks:
            print(i)
            if num == 0:
                temp = stock_price(i, period, traday[0], traday[-1])
                dataf = temp.copy()
                Alpha = Alphas(dataf)
                col_name = factor
                df_m = copy.deepcopy(dataf)
                df_m = df_m[['tradedate']]
                df_m['stockcode'] = i
                df_m[col_name] = eval(factor)()
                df_stk300 = df_m
            else:
                temp = stock_price(i, period, traday[0], traday[-1])
                dataf = temp.copy()
                Alpha = Alphas(dataf)
                col_name = factor
                df_m = copy.deepcopy(dataf)
                df_m = df_m[['tradedate']]
                df_m['stockcode'] = i
                df_m[col_name] = eval(factor)()
                df_stk300 = pd.concat([df_stk300, df_m])
            num = num + 1
        df_stk300 = df_stk300.merge(indust_stocks, on='stockcode')
        df_stk300.tradedate = df_stk300.tradedate.apply(lambda s: str(s)[:10])

        ret = pd.DataFrame()
        num = 0
        for idx, group in df_stk300.groupby('tradedate'):
            print(idx)
            if num == 0:
                try:
                    temp = group.copy()
                    temp = temp[['stockcode', factor, 'indust']]
                    total = len(temp)
                    temp = temp.assign(Alpha_rank=100 * (total - temp[factor].rank(ascending=False) + 1) / total)
                    factor_indust = temp[['indust', 'Alpha_rank']].groupby('indust').mean().reset_index()
                    factor_indust['date'] = idx
                    ret = factor_indust
                except:
                    print(idx + 'is not full')
            else:
                try:
                    temp = group.copy()
                    temp = temp[['stockcode', factor, 'indust']]
                    total = len(temp)
                    temp = temp.assign(Alpha_rank=100 * (total - temp[factor].rank(ascending=False) + 1) / total)
                    factor_indust = temp[['indust', 'Alpha_rank']].groupby('indust').mean().reset_index()
                    factor_indust['date'] = idx
                    ret = pd.concat([ret, factor_indust])
                except:
                    print(idx + 'is not full')
            num = num + 1

        df_300 = pd.DataFrame(stocks, columns=['stockcode'])
        temp = df_300.merge(indust_stocks).groupby('indust').count().reset_index()
        temp.stockcode = temp.stockcode.apply(lambda s: 1 if s > 3 else 0.5 if s < 2 else 0.9)
        temp = temp.iloc[:, :2]
        temp.columns = ['indust', 'q']
        ret_1 = ret.merge(temp)
        ret_1['score'] = ret_1['Alpha_rank'] * ret_1['q']
        ret_1['indust_c'] = ret_1['indust'].map(code_to_name)
        ret_1 = ret_1.dropna()

        num = 0
        ret_3 = pd.DataFrame()
        for idx, group in ret_1.groupby('date'):
            print(idx)
            if num == 0:
                temp = group.sort_values(by='score').iloc[-3:, :]
                temp['date'] = idx
                ret_3 = temp
            else:
                temp = group.sort_values(by='score').iloc[-3:, :]
                temp['date'] = idx
                ret_3 = pd.concat([ret_3, temp])
            num = num + 1

        stk_re = list()
        for idx, group in df_stk300.groupby('tradedate'):
            df = group.copy()
            indu_r = ret_3.query("date=='{var}'".format(var=idx)).indust.tolist()
            for i in indu_r:
                re = list()
                temp = df.query("indust=='{var}'".format(var=i))
                re.append(idx)
                re.append(i)
                re.append(temp.sort_values(by=factor).tail(2).stockcode.tolist())
                stk_re.append(re)

        stk_record = pd.DataFrame(stk_re)
        stk_record.columns = ['date', 'indust', 'stk_rec']
        stk_record['indust'] = stk_record['indust'].map(code_to_name)

        date = traday[-1][:10]

        stk_record = stk_record.sort_values(by='date', ascending=False)
        stk_record = stk_record.iloc[3:, :]
        stk_record.date = stk_record.date.shift(3)
        stk_record = stk_record.fillna(date)
        stk_record['stk_rec'] = [toname(row, stk_code_to_name) for idx, row in stk_record.iterrows()]
        stk_today = stk_record['indust'].tolist()
        stk_record.columns = ['日期', '行业', '参考']
        stk_record.index = range(1, len(stk_record) + 1)

        # =============================================================================
        # 发邮件
        # =============================================================================
        try:
            smtpserver = 'smtp.163.com'
            username = 'xiahutao@163.com'
            password = 'GQYJMWRAYELIJGWK'
            sender = 'xiahutao@163.com'
            # receiver='XXX@126.com'
            # 收件人为多个收件人
            #        receiver=['542362275@qq.com','43521385@qq.com','519518384@qq.com','xiaoliang.jia@jrj.com.cn','rdfce6@126.com','zxdokok@sina.com','654497127@qq.com','1518139212@qq.com','18244689637@163.com','kakaiphoimo@163.com','419393571@qq.com','554108967@qq.com','yanghui_hu@sina.com']
            # receiver=['542362275@qq.com','43521385@qq.com','zxdokok@sina.com','3467518502@qq.com', 'lfp-1203@163.com']
            # receiver = ['542362275@qq.com', '3467518502@qq.com', 'lfp-1203@163.com']
            receiver = ['542362275@qq.com', 'xiahutao@163.com']

            subject = date + ' ' + stk_today[0] + ',' + stk_today[1] + ',' + stk_today[2]
            # 通过Header对象编码的文本，包含utf-8编码信息和Base64编码信息。以下中文名测试ok
            # subject = '中文标题'
            # subject=Header(subject, 'utf-8').encode()

            # 构造邮件对象MIMEMultipart对象
            # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = 'xiahutao@163.com <xiahutao@163.com>'
            # msg['To'] = 'XXX@126.com'
            # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
            msg['To'] = ";".join(receiver)
            # msg['Date']='2012-3-16'

            # 构造文字内容
            text = '行业'
            text_plain = MIMEText(text, 'plain', 'utf-8')
            msg.attach(text_plain)
            html_msg = get_html_msg(stk_record)
            content_html = MIMEText(html_msg, "html", "utf-8")
            msg.attach(content_html)

            # 发送邮件
            smtp = smtplib.SMTP()
            smtp.connect('smtp.163.com')
            smtp.login(username, password)
            smtp.sendmail(sender, receiver, msg.as_string())
            smtp.quit()
        #    print('完成邮件发送')
        except Exception as e:
            print(e)
            # qq邮箱smtp服务器
            host_server = 'smtp.qq.com'
            # sender_qq为发件人的qq号码
            sender_qq = '542362275'
            # pwd为qq邮箱的授权码
            pwd = 'aacugluuqsfbbbaj'
            # 发件人的邮箱
            sender_qq_mail = '542362275@qq.com'
            # 收件人邮箱
            # receiver = ['542362275@qq.com', '43521385@qq.com', 'zxdokok@sina.com', '3467518502@qq.com']
            receiver = ['542362275@qq.com', '3467518502@qq.com']

            today = datetime.datetime.now()
            today = today.strftime('%Y-%m-%d')
            date = today
            # context1=res.iloc[0,:]['sign_position']
            subject = date + ' ' + stk_today[0] + ',' + stk_today[1] + ',' + stk_today[2]

            # 构造文字内容
            text = ''
            #        text_plain = MIMEText(text,'plain', 'utf-8')
            #        msg.attach(text_plain)

            # 邮件的正文内容
            mail_content = text
            # 邮件标题
            mail_title = subject
            # ssl登录
            smtp = SMTP_SSL(host_server)
            # set_debuglevel()是用来调试的。参数值为1表示开启调试模式，参数值为0关闭调试模式
            smtp.set_debuglevel(1)
            smtp.ehlo(host_server)
            smtp.login(sender_qq, pwd)

            msg = MIMEText(mail_content, "plain", 'utf-8')

            html_msg = get_html_msg(stk_record)
            content_html = MIMEText(html_msg, "html", "utf-8")
            msg.attach(content_html)

            msg["Subject"] = Header(mail_title, 'utf-8')
            msg["From"] = sender_qq_mail
            msg["To"] = ";".join(receiver)
            smtp.sendmail(sender_qq_mail, receiver, msg.as_string())
            smtp.quit()
