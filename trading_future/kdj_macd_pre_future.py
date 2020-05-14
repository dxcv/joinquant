# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
# from backtest_func import *
import os
import time
import matplotlib.pyplot as plt
from matplotlib import style
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import copy
import numpy as np
import pandas as pd
import talib
import tkinter
import tkinter.messagebox
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from analysis.report.graphs import Graphs
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph,NextPageTemplate,PageBreak,PageBegin
from reportlab.lib.pagesizes import letter
from data_engine.instrument.future import Future
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph,Spacer,Image,Table
from reportlab.lib.units import cm
import datetime
from reportlab.pdfgen import canvas
from reportlab.pdfgen.canvas import Canvas

from common.os_func import check_fold
pdfmetrics.registerFont(TTFont("SimSun", "G:/trading/SimSun.ttf"))

# auth('18610039264', 'zg19491001')
style.use('ggplot')

auth('15658001226', 'taiyi123')
myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
jzmongo = Arctic(myclient)



# 获取价格
def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    q = query(finance.GLOBAL_IDX_DAILY).filter(finance.GLOBAL_IDX_DAILY.code == sec).order_by(
        finance.GLOBAL_IDX_DAILY.day.desc())
    temp = finance.run_query(q)[
        ['day', 'name', 'code', 'open', 'high', 'low', 'close', 'volume']] \
        .assign(day=lambda df: df.day.apply(lambda x: str(x)[:10])) \
        .rename(columns={'day': 'trade_date', 'code': 'stock_cpde'})
    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)].sort_values(['trade_date'])
    return temp


def get_stock_code_list():
    db_index = jzmongo['stock_raw.stock_index']
    stock_df = db_index.read('all')
    code_list = list(stock_df.iloc[-1].dropna().index)
    return code_list


def stock_price_cgo(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    if sec in ['CU8888.XSGE', 'SN8888.XSGE', 'PB8888.XSGE', 'NI8888.XSGE', 'AL8888.XSGE', 'AU8888.XSGE', 'ZN8888.XSGE',
               'SC8888.XINE', 'AG8888.XSGE']:
        if sec == 'CU8888.XSGE':
            symble = 'HG'
        elif sec == 'ZN8888.XSGE':
            symble = 'ZSD'
        elif sec == 'SN8888.XSGE':
            symble = 'SND'
        elif sec == 'PB8888.XSGE':
            symble = 'PBD'
        elif sec == 'NI8888.XSGE':
            symble = 'NID'
        elif sec == 'AL8888.XSGE':
            symble = 'AHD'
        elif sec == 'SC8888.XINE':
            symble = 'CL'
        elif sec == 'AG8888.XSGE':
            symble = 'SI'
        else:
            symble = 'GC'
        temp = finance.run_query(query(finance.FUT_GLOBAL_DAILY).filter(
            finance.FUT_GLOBAL_DAILY.code == symble, finance.FUT_GLOBAL_DAILY.day >= sday,
            finance.FUT_GLOBAL_DAILY.day <= eday))
        temp = temp\
            .rename(columns={'day': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))[
            ['open', 'high', 'low', 'close', 'date_time', 'volume']]

    else:
        temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=True, fq='pre',
                         count=None).reset_index() \
            .rename(columns={'index': 'date_time'}) \
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

    temp['stock_code'] = sec
    return temp


def trans_heng_float(x):
    if x == '--':
        x = None
    return x


def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.stock_1d_jq_post'].read(sec)
    temp = temp[temp['volume'] > 0]
    temp['date_time'] = temp.index
    temp = temp.assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))

    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x))) \
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x))) \
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[
        ['high', 'open', 'low', 'close', 'date_time']].dropna()
    temp = temp[(temp['date_time'] >= sday) & (temp['date_time'] <= eday)].sort_values(['date_time'])

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
    return temp


def KDJ(data, N=9, M1=3, M2=3):
    datelen = len(data)
    data = data[['date_time', 'open', 'high', 'low', 'close']]
    array = np.array(data)
    kdjarr = []
    k_lst = []
    d_lst = []
    j_lst = []

    for i in range(datelen):
        if i - N < 0:
            b = 0
        else:
            b = i - N + 1
        rsvarr = array[b:i + 1, 0:5]
        try:
            rsv = (float(rsvarr[-1, -1]) - float(min(rsvarr[:, 3]))) / (
                    float(max(rsvarr[:, 2])) - float(min(rsvarr[:, 3]))) * 100
            if i == 0:
                k = rsv
                d = rsv
            else:
                k = 1 / float(M1) * rsv + (float(M1) - 1) / M1 * float(kdjarr[-1][2])
                d = 1 / float(M2) * k + (float(M2) - 1) / M2 * float(kdjarr[-1][3])
            j = 3 * k - 2 * d
        except Exception as e:
            k = 50
            d = 50
            j = 50
            rsv = 50
        k_lst.append(k)
        d_lst.append(d)
        j_lst.append(j)
        kdjarr.append(list((rsvarr[-1, 0], rsv, k, d, j)))

    return k_lst, d_lst, j_lst


def get_alert_info(df, txt):
    if len(df) > 0:
        info_txt = txt
        for idx, row in df_day_buy.iterrows():
            info_txt = info_txt + row.stock_code[:6] + ': k ' + str(np.around(row.k_day, 2)) + ' d ' + \
                       str(np.around(row.d_day, 2)) + ';'
        print(info_txt)
        tkinter.messagebox.showinfo('提示', info_txt)


def get_normal_future_index_code():
    temp = get_all_securities(types=['futures'])
    temp['index_code'] = temp.index
    temp['idx'] = temp['index_code'].apply(lambda x: x[-9:-5])
    temp = temp[temp['idx'] == '8888']
    temp['symbol'] = temp['index_code'].apply(lambda x: x[:-9])
    code_lst = temp.symbol.tolist()
    temp = temp[['index_code', 'symbol']].set_index(['symbol'])
    code_dic = {}
    for idx, _row in temp.iterrows():
        code_dic[idx] = _row.index_code

    return code_dic, code_lst


class PDFGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.file_path = 'G:/trading/'
        self.title_style = ParagraphStyle(name="TitleStyle", fontSize=48, alignment=TA_LEFT,)
        self.sub_title_style = ParagraphStyle(name="SubTitleStyle", fontSize=32,
                                              textColor=colors.HexColor(0x666666), alignment=TA_LEFT, )
        self.content_style = ParagraphStyle(name="ContentStyle", fontSize=18, leading=25, spaceAfter=20,
                                            underlineWidth=1, alignment=TA_LEFT, )
        self.foot_style = ParagraphStyle(name="FootStyle", fontSize=14, textColor=colors.HexColor(0xB4B4B4),
                                         leading=25, spaceAfter=20, alignment=TA_CENTER, )
        self.table_title_style = ParagraphStyle(name="TableTitleStyle", fontSize=20, leading=25,
                                                spaceAfter=10, alignment=TA_LEFT, )
        self.sub_table_style = ParagraphStyle(name="SubTableTitleStyle", fontSize=16, leading=25,
                                                spaceAfter=10, alignment=TA_LEFT, )
        self.basic_style = TableStyle([('FONTNAME', (0, 0), (-1, -1), 'ping'),
                                       ('FONTSIZE', (0, 0), (-1, -1), 12),
                                       ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                       ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                       # 'SPAN' (列,行)坐标
                                       ('SPAN', (1, 0), (3, 0)),
                                       ('SPAN', (1, 1), (3, 1)),
                                       ('SPAN', (1, 2), (3, 2)),
                                       ('SPAN', (1, 5), (3, 5)),
                                       ('SPAN', (1, 6), (3, 6)),
                                       ('SPAN', (1, 7), (3, 7)),
                                       ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                       ])
        self.common_style = TableStyle([('FONTNAME', (0, 0), (-1, -1), 'ping'),
                                      ('FONTSIZE', (0, 0), (-1, -1), 12),
                                      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                      ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                      ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                                     ])

    def genTaskPDF(self, hold_buy_day, hold_sell_day, hold_buy_week, hold_sell_week, other_buy_week, other_sell_week, method):
        styles = getSampleStyleSheet()
        normalStyle = copy.deepcopy(styles['Normal'])
        normalStyle.fontName = 'SimSun'
        story = []
        story.append(
            Graphs.draw_title('kd_macd_future_%s' % datetime.datetime.now().strftime('%Y%m%d')))
        story.append(Spacer(0, 0.5 * cm))

        story.append(Paragraph('在持仓日频KD金叉: ', normalStyle))
        story.append(Spacer(0, 0.2 * cm))
        data = [tuple(hold_buy_day.columns)] + [tuple(x.to_dict().values()) for idx, x in hold_buy_day.iterrows()]
        story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                         col_width=[80] + [70] * (len(hold_buy_day.columns) - 1)))
        story.append(Spacer(0, 0.5 * cm))

        story.append(Paragraph('在持仓日频KD死叉: ', normalStyle))
        story.append(Spacer(0, 0.2 * cm))
        data = [tuple(hold_sell_day.columns)] + [tuple(x.to_dict().values()) for idx, x in hold_sell_day.iterrows()]
        story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                       col_width=[80] + [70] * (len(hold_sell_day.columns) - 1)))
        story.append(Spacer(0, 0.5 * cm))

        if method == 1:
            story.append(Paragraph('在持仓周频KD金叉: ', normalStyle))
            story.append(Spacer(0, 0.2 * cm))
            data = [tuple(hold_buy_week.columns)] + [tuple(x.to_dict().values()) for idx, x in hold_buy_week.iterrows()]
            story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                           col_width=[80] + [70] * (len(hold_buy_week.columns) - 1)))
            story.append(Spacer(0, 0.5 * cm))

            story.append(Paragraph('在持仓周频KD死叉: ', normalStyle))
            story.append(Spacer(0, 0.2 * cm))
            data = [tuple(hold_sell_week.columns)] + [tuple(x.to_dict().values()) for idx, x in hold_sell_week.iterrows()]
            story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                           col_width=[80] + [70] * (len(hold_sell_week.columns) - 1)))
            story.append(Spacer(0, 0.5 * cm))

            story.append(Paragraph('非在持仓周频KD金叉: ', normalStyle))
            story.append(Spacer(0, 0.2 * cm))
            data = [tuple(other_buy_week.columns)] + [tuple(x.to_dict().values()) for idx, x in other_buy_week.iterrows()]
            story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                           col_width=[80] + [70] * (len(other_buy_week.columns) - 1)))
            story.append(Spacer(0, 0.5 * cm))

            story.append(Paragraph('非在持仓周频KD死叉: ', normalStyle))
            story.append(Spacer(0, 0.2 * cm))
            data = [tuple(other_sell_week.columns)] + [tuple(x.to_dict().values()) for idx, x in
                                                      other_sell_week.iterrows()]
            story.append(Graphs.draw_table(*data, ALIGN='LEFT', VALIGN='RIGHT',
                                           col_width=[80] + [70] * (len(other_sell_week.columns) - 1)))
            story.append(Spacer(0, 0.5 * cm))

        doc = SimpleDocTemplate(self.file_path + self.filename + ".pdf", pagesize=letter)
        doc.build(story)


def get_date(calen, today):
    next_tradeday = get_trade_days(start_date=today + datetime.timedelta(days=1), end_date='2030-01-01')[0]
    if datetime.datetime.now().hour > 18:
        calen.append(next_tradeday)
    EndDate = calen[-1]
    StartDate = calen[0]
    listDATE = calen[-125]
    hq_last_date = calen[-2]
    return calen, next_tradeday, EndDate, listDATE, StartDate, str(hq_last_date)[:10]


if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    t0 = time.time()
    fold = 'e:/kdj_macd/'
    fold_data = 'e:/kdj_macd/data/'
    # data = get_all_securities(types=['futures'])

    start_day = '2017-01-01'
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    calen = get_trade_days(count=3)
    calen = list(calen)
    if datetime.datetime.now().hour < 9:
        end_day = calen[-2].strftime('%Y-%m-%d')
    # end_day = '2020-04-20'
    mod = 0
    hold_code_lst = ['RU', 'P', 'AU', 'C', 'CU', 'ZN']
    normalize_code_future, index_code_lst = get_normal_future_index_code()
    method = 1
    n = 1  # 回测周期
    period = '1d'
    method_lst = [('week', 'week'), ('day', 'day')]
    k1_lst = [(20, 30)]  # kd下限
    k2_lst = [(70, 80)]  # kd上限
    if mod == 0:
        k1 = (15, 35)
        k2 = (65, 85)
    else:
        k1 = (20, 30)
        k2 = (70, 80)
    df_lst = []
    hq_dict = {}
    vol_dict = {}
    for index_code in index_code_lst:
        symble = normalize_code_future[index_code]
        data_daily = stock_price_cgo(symble, start_day, end_day)[
            ['date_time', 'open', 'high', 'low', 'close', 'stock_code', 'volume']]
        data_daily['vol_average'] = data_daily['volume'].shift(1).rolling(window=30).mean()
        print(data_daily)
        if len(data_daily) < 30:
            continue
        hq_dict[index_code] = data_daily
        vol_dict[index_code] = [data_daily.vol_average.tolist()[-1]]
    vol_df = pd.DataFrame(vol_dict)
    print(vol_df)
    vol_df = vol_df.T
    vol_df.columns = ['volume_ave']
    print(len(vol_df))
    vol_df = vol_df[vol_df['volume_ave'] >= 50000]
    print(vol_df)
    index_code_lst = vol_df.index.values
    index_code_lst = list(index_code_lst)
    index_code_lst.extend(hold_code_lst)
    index_code_lst = set(index_code_lst)

    for index_code in index_code_lst:
        data_daily = hq_dict[index_code][
            ['date_time', 'open', 'high', 'low', 'close', 'stock_code']]
        data_daily['time'] = pd.to_datetime(data_daily['date_time'])
        data_daily.index = data_daily['time']
        data_daily = data_daily.drop(['time'], axis=1)
        data_week = data_daily.resample('W').last()
        data_week['date_time'] = data_daily['date_time'].resample('W').last()
        data_week['open'] = data_daily['open'].resample('W').first()
        data_week['high'] = data_daily['high'].resample('W').max()
        data_week['low'] = data_daily['low'].resample('W').min()
        data_week['close'] = data_daily['close'].resample('W').last()
        data_week = data_week.dropna()
        data_week['k_week'], data_week['d_week'], data_week['j_week'] = KDJ(data_week, 9, 3, 3)

        data_week['MACD'], data_week['MACDsignal'], data_week['macd_week'] = talib.MACDEXT(
            data_week['close'].values, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9,
            signalmatype=1)
        data_week = data_week.drop(['MACD', 'MACDsignal'], axis=1)

        data_daily['k_day'], data_daily['d_day'], data_daily['j_day'] = KDJ(data_daily, 9, 3, 3)
        data_daily['MACD'], data_daily['MACDsignal'], data_daily['macd_day'] = talib.MACDEXT(
            data_daily['close'].values, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1, signalperiod=9,
            signalmatype=1)
        data_daily = data_daily.drop(['MACD', 'MACDsignal'], axis=1)

        data_daily = data_daily.merge(
            data_week[['k_week', 'd_week', 'macd_week', 'date_time']], on=['date_time'], how='left').sort_values(
            ['date_time'])
        data_daily = data_daily.fillna(method='ffill')
        # data_daily.to_csv(fold_data + symble + '_data_daily.csv')
        data_daily['day_kd_b'] = (data_daily['k_day'] > data_daily['d_day']) & (
                data_daily['k_day'].shift(1) < data_daily['d_day'].shift(1)) & (
                                         data_daily['k_day'] < k1[1]) & (data_daily['d_day'] > k1[0])
        data_daily['day_kd_s'] = (data_daily['k_day'] < data_daily['d_day']) & (
                data_daily['k_day'].shift(1) > data_daily['d_day'].shift(1)) & (
                                         data_daily['k_day'] > k2[0]) & (data_daily['d_day'] < k2[1])
        data_daily['week_kd_b'] = (data_daily['k_week'] > data_daily['d_week']) & (
                data_daily['k_week'].shift(1) < data_daily['d_week'].shift(1)) & (
                                          data_daily['k_week'] < k1[1]) & (data_daily['d_week'] > k1[0])
        data_daily['week_kd_s'] = (data_daily['k_week'] < data_daily['d_week']) & (
                data_daily['k_week'].shift(1) > data_daily['d_week'].shift(1)) & (
                                          data_daily['k_week'] > k2[0]) & (data_daily['d_week'] < k2[1])
        data_daily = data_daily.dropna()
        data_daily['macd_week_abs'] = np.abs(data_daily['macd_week'])
        data_daily['rank_week'] = data_daily.macd_week_abs.rank(method='min').astype(int)
        data_daily['percentile_week'] = data_daily['rank_week'] / len(data_daily)
        data_daily['macd_day_abs'] = np.abs(data_daily['macd_day'])
        data_daily['rank_day'] = data_daily.macd_day_abs.rank(method='min').astype(int)
        data_daily['percentile_day'] = data_daily['rank_day'] / len(data_daily)
        df_lst.append(data_daily.tail(1))
    df = pd.concat(df_lst)

    name_lst = []
    for code in df.stock_code.tolist():
        name_lst.append(get_security_info(code).display_name)
    df['简称'] = name_lst
    df[['k_day', 'd_day', 'k_week', 'd_week', 'percentile_week', 'percentile_day']] = df[
        ['k_day', 'd_day', 'k_week', 'd_week', 'percentile_week', 'percentile_day']].apply(lambda x: np.around(x, 2))
    df['stock_code'] = df['stock_code'].apply(lambda x: x[:-9])
    df = df[['date_time', '简称', 'stock_code', 'day_kd_b', 'day_kd_s', 'week_kd_b', 'week_kd_s', 'k_day', 'd_day', 'k_week',
             'd_week', 'percentile_day', 'percentile_week']]

    df_day_buy = df[df['day_kd_b']==True]
    df_day_sell = df[df['day_kd_s']==True]
    df_week_buy = df[df['week_kd_b']==True]
    df_week_sell = df[df['week_kd_s']==True]
    # get_alert_info(df_day_buy, '日级别KD金叉:')
    # get_alert_info(df_week_buy, '周级别KD金叉:')
    # get_alert_info(df_day_sell, '日级别KD死叉:')
    # get_alert_info(df_week_sell, '周级别KD死叉:')


    df.columns = ['日期', '简称', '代码', '日金叉', '日死叉', '周金叉', '周死叉', '日K', '日D', '周K', '周D', 'MACD日分位点', 'MACD周分位点']
    df['简称'] = df['简称'].apply(lambda x: x[:-4])
    print(df[['日金叉', '代码', '日死叉', '周金叉', '周死叉', '日K', '日D', '周K', '周D', 'MACD日分位点', 'MACD周分位点']])

    df_day_buy = df[df['日金叉'] == True][['日期', '简称', '代码', '日K', '日D', 'MACD日分位点']]
    df_day_sell = df[df['日死叉'] == True][['日期', '简称', '代码', '日K', '日D', 'MACD日分位点']]
    df_week_buy = df[df['周金叉'] == True][['日期', '简称', '代码', '周K', '周D', 'MACD周分位点']]
    df_week_sell = df[df['周死叉'] == True][['日期', '简称', '代码', '周K', '周D', 'MACD周分位点']]

    df_day_buy_hold = df_day_buy[df_day_buy['代码'].isin(hold_code_lst)]
    df_day_sell_hold = df_day_sell[df_day_sell['代码'].isin(hold_code_lst)]

    df_week_buy_hold = df_week_buy[df_week_buy['代码'].isin(hold_code_lst)]
    df_week_sell_hold = df_week_sell[df_week_sell['代码'].isin(hold_code_lst)]

    other_code = [i for i in index_code_lst if i not in hold_code_lst]

    print('持仓日频KD金叉标的：===========================')
    print(df_day_buy_hold)
    print('持仓周频KD金叉标的：===========================')
    print(df_week_buy_hold)
    print('持仓日频KD死叉标的：===========================')
    print(df_day_sell_hold)
    print('持仓周频KD死叉标的：===========================')
    print(df_week_sell_hold)

    print('非持仓周频KD金叉标的：===========================')
    print(df_week_buy[df_week_buy['代码'].isin(other_code)])
    print('非持仓周频KD死叉标的：===========================')
    print(df_week_sell[df_week_sell['代码'].isin(other_code)])


    df.to_csv(fold_data + 'kdj_future_' + end_day + '1.csv', encoding='gbk')
    print(time.time() - t0)
    PDFGenerator('kdj_macd_future_signal_' + end_day).genTaskPDF(
        df_day_buy_hold, df_day_sell_hold, df_week_buy_hold, df_week_sell_hold,
        df_week_buy[df_week_buy['代码'].isin(other_code)], df_week_sell[df_week_sell['代码'].isin(other_code)],
               method)
