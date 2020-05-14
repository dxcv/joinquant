# coding:UTF-8
'''
Created on Jun 13, 2016

@author: peng
'''
from __future__ import division
import pandas as pd
import numpy as np
import math
import time
from joinquant import *
import matplotlib.pyplot as plt
import datetime
from jqdatasdk import *
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from smtplib import SMTP_SSL

auth('18610039264', 'zg19491001')


# from mssqlDBConnectorPeng import *


# from ReadIndex import *


def dealindexdata(indexdata, startDate, endDate, prenum, afternum):
    indexdata = indexdata[startDate:endDate]
    timelistraw = list(indexdata['TRADEDATE'])
    pricelistraw = np.float64(list(indexdata['TCLOSE']))
    # daypctlistraw = np.float64(list(indexdata['CHNG_PCT']))

    datalist = []
    labellist = []
    datelist = []
    periodlist = []
    daypctlist = []
    for i in range(len(pricelistraw) - prenum + 1):
        if i < len(pricelistraw) - prenum - afternum + 1:

            datalisttmp = []
            for j in range(prenum):
                datalisttmp.append(pricelistraw[i + j])
            datalist.append(datalisttmp)
            tmp1 = pricelistraw[i + prenum - 1]
            tmp2 = pricelistraw[i + prenum + afternum - 1]
            tmp = (tmp2 - tmp1) / tmp1

            labellist.append(tmp)

            datelist.append(timelistraw[i + prenum - 1])

            timetmp = timelistraw[i] + '~' + timelistraw[i + prenum - 1]
            periodlist.append(timetmp)

            daypctlist.append(
                (pricelistraw[i + prenum - 1] - pricelistraw[i + prenum - 2]) / pricelistraw[i + prenum - 2] * 100.0)
        if i >= len(pricelistraw) - prenum - afternum + 1:
            datalisttmp = []
            for j in range(prenum):
                datalisttmp.append(pricelistraw[i + j])
            datalist.append(datalisttmp)

            labellist.append(None)

            datelist.append(timelistraw[i + prenum - 1])

            timetmp = timelistraw[i] + '~' + timelistraw[i + prenum - 1]
            periodlist.append(timetmp)

            daypctlist.append(
                (pricelistraw[i + prenum - 1] - pricelistraw[i + prenum - 2]) / pricelistraw[i + prenum - 2] * 100.0)
    return datalist, labellist, datelist, periodlist, daypctlist


def normalize(list):
    '''
    normalize the data
    input parameter list
    '''
    listtmp = []
    a = np.float64(max(list))
    for i in range(len(list)):
        c = (np.float64(list[i])) / a
        listtmp.append(c)
    return listtmp


def caldistancefactor(x, y):
    '''
    :param array x: N1*1 array
    :param array y: N2*1 array
    '''
    dis = np.float64(0)
    # cc = np.float64(0)
    for a, b in zip(x, y):
        dis = dis + math.fabs(a - b)
        # cc = cc + (a-b)*(a-b)
    # caltmp = math.sqrt(np.float64(len(x)))*math.sqrt(cc)
    return 1 - dis


def datanorm(datalist):
    datatmp = []
    for i in range(len(datalist)):
        listtmp = normalize(datalist[i])
        datatmp.append(listtmp)
    return datatmp


def select_similarity(list, listall, method, param):
    '''
    method: 1. Manhattan Distance method;
    param: similarity measurement
    '''
    corlist = []
    indexlist = []
    if method == 1:
        for i in range(len(listall)):
            r = caldistancefactor(list, listall[i])
            if r > param:
                indexlist.append(i)
            corlist.append(r)
    else:
        for i in range(len(listall)):
            r = caldistancefactor(list, listall[i])
            if r > param:
                indexlist.append(i)
            corlist.append(r)
    return corlist, indexlist


def predict(indexlist, trainlabellist, prob):
    pre = 0
    above = 0
    below = 0
    held = 0
    if len(indexlist) >= 5:
        for i in indexlist:
            if trainlabellist[i] > 0:
                above = above + 1
            elif trainlabellist[i] < 0:
                below = below + 1
            else:
                held = held + 1
        if np.float64(above) / np.float64(len(indexlist)) > prob:
            pre = 1
        elif np.float64(below) / np.float64(len(indexlist)) > prob:
            pre = -1
        print((above, below, np.float64(above) / np.float64(len(indexlist)), np.float64(below) / np.float64(
            len(indexlist))))
    else:
        pre = 0
    return pre


def selectdata(datalist, labellist, datelist, periodlist, daypctlist, startDate, endDate):
    datedf = pd.DataFrame(datelist, columns=['date'])
    datedf = datedf.set_index('date', drop=False)
    selectdate = list(datedf[startDate:endDate]['date'])
    startindex = datelist.index(selectdate[0])
    endindex = datelist.index(selectdate[-1])
    datalist = datalist[:endindex + 1]
    labellist = labellist[:endindex + 1]
    datelist = datelist[:endindex + 1]
    periodlist = periodlist[:endindex + 1]
    daypctlist = daypctlist[:endindex + 1]
    return datalist, labellist, datelist, periodlist, daypctlist, startindex


def testpredict(datalist, labellist, datelist, periodlist, daypctlist, afternum, dnumber, method,
                similarity=0.7,
                prob=0.55):
    positive = 0
    negtive = 0
    prelist = []
    profitlist = []
    longprofitlist = []
    ####################################
    for n in range(len(datalist) - dnumber):
        corlist, indexlist = select_similarity(datalist[n + dnumber], datalist[0:dnumber + n - afternum + 1], method,
                                               similarity)
        # print len(indexlist), '\n', indexlist
        trainlabellist = labellist[0:dnumber + n - afternum + 1]
        pre = predict(indexlist, trainlabellist, prob)
        print('评估期限：%s 相似曲线数量：%s 预测涨跌：%s 实际涨跌：%s' % (
            periodlist[n + dnumber], len(indexlist), pre, labellist[n + dnumber]))
        prelist.append(pre)

        lsprofit = np.float64(pre) * np.float64(labellist[n + dnumber])
        profitlist.append(lsprofit)
        if pre == 1:
            longprofitlist.append(lsprofit)
        else:
            longprofitlist.append(np.float64(0.0))

        if lsprofit > 0.0:
            positive = positive + 1
        elif lsprofit < 0:
            negtive = negtive + 1

    positive = np.float64(positive)
    negtive = np.float64(negtive)
    acc = positive / (positive + negtive)
    freq = (positive + negtive) / np.float64(len(datalist) - dnumber)

    holdlist, poslist = countpositions(prelist, afternum)
    resultdf = pd.DataFrame(datelist[dnumber:], columns=['Date']) \
        .assign(Period=periodlist[dnumber:]) \
        .assign(Predict=prelist) \
        .assign(Position=poslist) \
        .assign(Label=labellist[dnumber:]) \
        .assign(Lsprofit=profitlist) \
        .assign(Longprofit=longprofitlist) \
        .assign(Pctchange=daypctlist[dnumber:]) \
        .assign(Hold=holdlist)
    # alldatalist = []
    # alldatalist.append(datelist[dnumber:])
    # alldatalist.append(periodlist[dnumber:])
    # alldatalist.append(prelist)
    # alldatalist.append(labellist[dnumber:])
    # alldatalist.append(profitlist)
    # alldatalist.append(longprofitlist)
    # alldatalist.append(daypctlist[dnumber:])
    # alldatalist.append(holdlist)
    # alldatalist = map(list, zip(*alldatalist))
    # print(len(alldatalist))
    # columntmp = ['Date', 'Period', 'Predict', 'Label', 'Lsprofit', 'Longprofit', 'Pctchange', 'Hold']
    # resultdf = pd.DataFrame(alldatalist, columns=columntmp)
    # resultdf = resultdf.set_index('Date', drop=False)

    return resultdf


def countpositions(prelist, n):
    for i in range(len(prelist)):
        if prelist[i] == -1:
            prelist[i] = 0
    listtmp = []
    for i in range(len(prelist)):
        listtmp.append(0)
    pos_lst = [0 for i in range(len(prelist))]

    for i in range(len(prelist)):
        if prelist[i] == 1:
            if i < len(prelist) - n:
                for j in range(n - 1):
                    prelist[i + j + 1] = 0
            if i >= len(prelist) - n and i < len(prelist) - 1:
                for j in range(len(prelist) - i - 1):
                    prelist[i + j + 1] = 0
    # print prelist
    for i in range(len(prelist)):
        if prelist[i] == 1:
            pos_lst[i] = 1
            if i < len(prelist) - n:
                for j in range(n):
                    listtmp[i + j + 1] = 1
                    pos_lst[i + j] = 1
            if i >= len(prelist) - n and i < len(prelist) - 1:
                for j in range(len(prelist) - i):
                    if i + j < len(prelist) - 1:
                        listtmp[i + j + 1] = 1
                    pos_lst[i + j] = 1

    return listtmp, pos_lst


def evaluate(df):
    lsprofitlist = list(df['Lsprofit'].dropna())
    longprofitlist = list(df['Longprofit'].dropna())
    daypctlist = list(df['Pctchange'])
    positonlist = list(df['Hold'])
    datelist = list(df['Date'])
    predict_lst = list(df['Position'])

    rightlist = []
    wronglist = []
    for i in lsprofitlist:
        if i > 0.0:
            rightlist.append(i)
        elif i < 0.0:
            wronglist.append(i)

    r = np.float64(len(rightlist))
    w = np.float64(len(wronglist))
    # 正确率
    acc = r / (r + w)
    # 盈亏比
    odd = -np.mean(rightlist) / np.mean(wronglist)
    # 预测频率
    freq = (r + w) / np.float64(len(lsprofitlist))

    # 考虑做多做空的收益率列表
    newlsprofitlist = []
    for i in lsprofitlist:
        if i != np.float64(0.0):
            newlsprofitlist.append(i)
    # 仅考虑做多的收益率列表
    newlongprofitlist = []
    for i in longprofitlist:
        if i != np.float64(0.0):
            newlongprofitlist.append(i)
    # 考虑做多做空的累积收益率

    lsprofitsum = sum(newlsprofitlist)
    # 考虑做多的累积收益率
    longprofitsum = sum(newlongprofitlist)

    # 考虑做多做空的复合收益率
    lsvalue = 1.0
    for i in newlsprofitlist:
        lsvalue = lsvalue * (1.0 + i)
    # 考虑做多的复合收益率
    longvalue = 1.0
    for i in newlongprofitlist:
        longvalue = longvalue * (1.0 + i)

    # 不同交易日资产净值列表（不考虑止损）
    realNAVlist = []
    # 不同交易日资产净值（不考虑止损）
    realNAV = 1.0
    for i in range(len(positonlist)):
        realNAV = realNAV * (1.0 + (positonlist[i] * (daypctlist[i] / 100)))
        realNAVlist.append(realNAV)

    indextrend = []
    # 预测标的实际走势
    indexNAV = 1.0
    for i in range(len(positonlist)):
        indexNAV = indexNAV * (1.0 + (daypctlist[i] / 100))
        indextrend.append(indexNAV)

    alldatalist = []
    alldatalist.append(datelist)
    alldatalist.append(daypctlist)
    alldatalist.append(positonlist)
    alldatalist.append(indextrend)
    alldatalist.append(realNAVlist)
    alldatalist.append(predict_lst)
    alldatalist = map(list, zip(*alldatalist))
    columnresult = ['Date', 'Pctchange', 'Hold', 'Indextrend', 'No_cuttingloss', 'position']
    result = pd.DataFrame(alldatalist, columns=columnresult)
    result = result.set_index('Date')
    return result, acc, odd, freq, lsprofitsum, longprofitsum, newlsprofitlist, newlongprofitlist


def maxRetrace(list):
    row = []
    for i in range(len(list)):
        row.append(1 - list[i] / max(list[:i + 1]))
    Max = max(row)
    return Max


def daysharpRatio(list):
    row = []
    for i in range(1, len(list)):
        row.append(math.log(list[i] / list[i - 1]))
    return np.mean(row) / np.std(row)


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


if __name__ == '__main__':
    # 进行研究的指数列表
    # indexcodelist = ['000001.XSHG', '000016.XSHG', '000300.XSHG', '000905.XSHG', '000906.XSHG']#上证指数、上证50、沪深300、中证500、中证800
    indexcode = '000001.XSHG'
    # indexcode = '000651.XSHE'
    today1 = datetime.date.today()
    print(today1)
    transferstr = lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')
    calen = get_trade_days(count=252)
    if today1 in calen:
        next_tradeday = get_trade_days(start_date=today1 + datetime.timedelta(days=1), end_date='2030-01-01')[0]
        calen = list(calen)
        if datetime.datetime.now().hour > 15:
            calen.append(next_tradeday)
        EndDate = calen[-1]
        # EndDate = '2020-03-31'
        today = EndDate

        today = datetime.datetime.strftime(today, '%Y-%m-%d')
        print(today)
        indexdata = stock_price(indexcode, '2004-01-01', calen[-2])
        # indexdata = stock_price(indexcode, '2004-01-01', EndDate)
        print(indexdata)
        '''
        storedData = pd.read_hdf('storedIndexPrices.h5','D20161001')
        print storedData
        
        indexdata = pd.concat([indexdata, storedData], ignore_index=True)
        #indexdata = indexdata.drop_duplicates()
        print indexdata
        '''

        indexdata = indexdata[['tradedate', 'close']] \
            .rename(columns={'tradedate': 'TRADEDATE', 'close': 'TCLOSE'})

        # indexdata = indexdata[indexdata['SECCODE'] == '000001']
        indexdata = indexdata.sort_values(by='TRADEDATE', ascending=True)
        transferstr = lambda x: datetime.datetime.strftime(x, '%Y-%m-%d')
        indexdata['TRADEDATE'] = indexdata['TRADEDATE'].apply(transferstr)
        indexdata = indexdata.set_index('TRADEDATE', drop=False)
        print(indexdata)

        # correlation coefficient method
        method = 1
        similarityset = 0.67
        probset = 0.67
        prenum = 20
        afternum = 3
        mod = 'future'

        t0 = time.time()
        datalist, labellist, datelist, periodlist, daypctlist = dealindexdata(indexdata, '2004-01-01', today, prenum,
                                                                              afternum)

        datalist = datanorm(datalist)

        datalist, labellist, datelist, periodlist, daypctlist, dnumber = selectdata(datalist, labellist, datelist,
                                                                                    periodlist, daypctlist,
                                                                                    '2006-01-01',
                                                                                    today)
        print(periodlist[-1])

        # dnumber = 2000

        resultdf = testpredict(datalist, labellist, datelist, periodlist, daypctlist, afternum, dnumber, method,
                               similarityset, probset)
        print(resultdf)
        result, acc, odd, freq, lsprofitsum, longprofitsum, lsprofitlist, longprofitlist = evaluate(resultdf)
        print(result)
        if result.position.tolist()[-1] == 0:
            ret = 'none'
        elif result.position.tolist()[-1] == 1:
            ret = 'long'
        elif result.position.tolist()[-1] == -1:
            ret = 'short'
        print('正确率：%s 盈亏比：%s 预测频率：%s 多空累积收益：%s 做多累积收益：%s' % (acc, odd, freq, lsprofitsum, longprofitsum))
        sharpRatio = daysharpRatio(result['No_cuttingloss'].tolist()) * math.pow(252, 0.5)
        annROR = math.pow(result['No_cuttingloss'].tolist()[-1], 252 / len(result)) - 1
        maxRetrac = maxRetrace(result['No_cuttingloss'].tolist())
        print('夏普比率= %s' % sharpRatio)
        print('年化收益率= %s' % annROR)
        print('最大历史回撤= %s' % maxRetrac)

        # =============================================================================
        # 将信息发送邮件
        # =============================================================================
        # 设置smtplib所需的参数
        # 下面的发件人，收件人是用于邮件传输的。
        smtpserver = 'smtp.163.com'
        username = 'xiahutao@163.com'
        password = 'GQYJMWRAYELIJGWK'
        sender = 'xiahutao@163.com'
        # receiver='XXX@126.com'
        # 收件人为多个收件人
        # receiver=['542362275@qq.com','43521385@qq.com','519518384@qq.com','xiaoliang.jia@jrj.com.cn','rdfce6@126.com','zxdokok@sina.com','654497127@qq.com','1518139212@qq.com','18244689637@163.com','kakaiphoimo@163.com','419393571@qq.com','554108967@qq.com','yanghui_hu@sina.com']
        receiver = ['542362275@qq.com', '3467518502@qq.com', 'xiahutao@163.com']
        # receiver = ['xiahutao@163.com', '3467518502@qq.com']

        date = today
        context = date + '今日适合:' + ret + '胜率：%s 盈亏比：%s 夏普：%s 年化收益：%s 最大回撤：%s' % (
            np.round(acc, 2), np.round(odd, 2), np.round(sharpRatio, 2), np.round(annROR, 2), np.round(maxRetrac, 2))

        res_n = result.copy().reset_index().loc[:, ['Date', 'position', 'No_cuttingloss']] \
            .dropna() \
            .sort_values(by='Date', ascending=False).head(100).reset_index(drop=True)
        res_n.Date = res_n.Date.shift(1)
        res_n.Date = res_n.Date.fillna(today)
        res_n.columns = ['日期', '建议', '净值']
        print(res_n)

        # subject = date + ' ' + context1
        subject = context
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
        text = subject
        text_plain = MIMEText(text, 'plain', 'utf-8')
        msg.attach(text_plain)
        html_msg = get_html_msg(res_n)
        content_html = MIMEText(html_msg, "html", "utf-8")
        msg.attach(content_html)

        # 发送邮件
        smtp = smtplib.SMTP()
        smtp.connect('smtp.163.com')
        smtp.login(username, password)
        smtp.sendmail(sender, receiver, msg.as_string())
        smtp.quit()
        #    print('完成邮件发送')

        # result.ix[:, ['Indextrend', 'No_cuttingloss']].plot()
        #
        # print('time spend : %s' % (time.time() - t0))
        # plt.show()
