# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 16:00:44 2018

@author: Administrator
"""

from joinquant import *
import datetime
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from smtplib import SMTP_SSL


def get_html_msg(data):
    """
    1. 构造html信息
    """
    df = data.copy()
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
            <h1 align='center'>情绪打分 </h1>\
        </div>\
        <hr>\
        <div class='content'>\
            <!--正文内容-->\
            <h2>情绪打分：</h2>\
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
    html_msg= "<html>" + head + body + "</html>"
    # 这里是将HTML文件输出，作为测试的时候，查看格式用的，正式脚本中可以注释掉
    fout = open('t4.html', 'w', encoding='UTF-8', newline='')
    fout.write(html_msg)
    return html_msg


if __name__ == "__main__":
    today = datetime.date.today()
    print(today)
    s_day = get_trade_days(count=30)[-20:]
    print(s_day)
    if today in s_day:
        count = 20
        code_lst300 = get_index_stocks('000300.XSHG', today)
        code_lst50 = get_index_stocks('000016.XSHG', today)
        code_lst300.extend(code_lst50)
        hq_data = []
        for code in code_lst300:
            print(code)
            one_data = get_stock_price(code, count, '1d', True, None, None)\
                .assign(diff=lambda df: (df.high - df.low) / df.low)
            hq_data.append(one_data)
        hq_data = pd.concat(hq_data)
        print(hq_data)
        hq_data = hq_data[hq_data['diff'] >= 0.02]
        diff_lst = []

        for code, group_m in hq_data.groupby(['stock_code']):
            group_w_2 = group_m[(group_m['diff'] >= 0.02) & (group_m['date_time'] >= s_day[-5])]
            group_d_2 = group_m[(group_m['diff'] >= 0.02) & (group_m['date_time'] >= s_day[-3])]

            group_m_3 = group_m[(group_m['diff'] >= 0.03)]
            group_w_3 = group_m[(group_m['diff'] >= 0.03) & (group_m['date_time'] >= s_day[-5])]
            group_d_3 = group_m[(group_m['diff'] >= 0.03) & (group_m['date_time'] >= s_day[-3])]

            group_m_5 = group_m[(group_m['diff'] >= 0.05)]
            group_w_5 = group_m[(group_m['diff'] >= 0.05) & (group_m['date_time'] >= s_day[-5])]
            group_d_5 = group_m[(group_m['diff'] >= 0.05) & (group_m['date_time'] >= s_day[-3])]
            diff_row = []
            diff_row.append(code)
            diff_row.append(len(group_m))
            diff_row.append(len(group_w_2))
            diff_row.append(len(group_d_2))
            diff_row.append(len(group_m_3))
            diff_row.append(len(group_w_3))
            diff_row.append(len(group_d_3))
            diff_row.append(len(group_m_5))
            diff_row.append(len(group_w_5))
            diff_row.append(len(group_d_5))
            diff_lst.append(diff_row)
        all_df = pd.DataFrame(
            diff_lst, columns=['stock_code', 'chng_m_2', 'chng_w_2', 'chng_d_2', 'chng_m_3', 'chng_w_3', 'chng_d_3',
                               'chng_m_5', 'chng_w_5', 'chng_d_5']) \
            .fillna(value=0).set_index(['stock_code'])
        all_df['sum_col'] = all_df.apply(lambda x: x.sum(), axis=1)
        print(all_df)
        all_stocks_name = get_all_stock_name()
        all_df = all_df.reset_index(drop=False)[['stock_code', 'sum_col']] \
            .sort_values(['sum_col'], ascending=[False]).head(50)\
            .rename(columns={'sum_col': 'score'}).merge(all_stocks_name, on=['stock_code'])
        smtpserver = 'smtp.163.com'
        username = 'xiahutao@163.com'
        password = 'yo425408'
        sender = 'xiahutao@163.com'
        receiver = ['519518384@qq.com', '3467518502@qq.com']
        subject = '沪深300活跃股打分'

        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = 'xiahutao@163.com <xiahutao@163.com>'
        # msg['To'] = 'XXX@126.com'
        # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
        msg['To'] = ";".join(receiver)
        # msg['Date']='2012-3-16'

        # 构造文字内容
        text = ''
        text_plain = MIMEText(text, 'plain', 'utf-8')
        msg.attach(text_plain)
        html_msg = get_html_msg(all_df)
        content_html = MIMEText(html_msg, "html", "utf-8")
        msg.attach(content_html)

        # 发送邮件
        smtp = smtplib.SMTP()
        smtp.connect(smtpserver)
        smtp.login(username, password)
        smtp.sendmail(sender, receiver, msg.as_string())
        smtp.quit()