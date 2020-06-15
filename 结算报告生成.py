#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import requests
from lxml import etree
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
total_asset = 7209753
myclient = pymongo.MongoClient('mongodb://dbmanager_stock:jz471042@192.168.2.201:27017/')
jzmongo = Arctic(myclient)

chrome_options = Options()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get("http://www.sse.com.cn/services/hkexsc/disclo/ratios/")
time.sleep(5)
exchange_rate = driver.find_element_by_xpath(
    '/html/body/div[8]/div[2]/div[2]/div[2]/div/div/div/div/div[2]/div[2]/div/div[2]/table/tbody/tr[2]/td[2]/div').text
exchange_rate = float(exchange_rate)
driver.close()

print('今日参考汇率: ' + str(exchange_rate))
def transfercode(x):
    x = str(int(x))
    if len(x)<6:
        x = '0' * (6-len(x)) + x
    return x
###设定 df是浙商账户 df2是海通账户 total_assetss是两个账户资产总计
df = pd.read_excel('g://trading//stock1.xlsx')
df2 = pd.read_excel('g://trading//持仓情况.xlsx', encoding='gbk').rename(columns={'最新价': '市价', '浮动盈亏': '盈亏'}).iloc[:-1, :]
print(df2)
df2['证券代码'] = df2['证券代码'].apply(lambda x: transfercode(x))
print(df2)

export_file_path = 'g://trading//账户持仓报告.xlsx'

df = df[:-1]
df2 = df2[:-1]



# 处理df2
temp = pd.DataFrame()
for stock_name in df2['证券名称']:
    print(stock_name)
    stock_code = str(df2[df2['证券名称'] == stock_name]['证券代码'].iloc[0])

    # 两个账户同时持有一只股票
    if stock_name in df['证券名称'].to_list():
        market1 = df[df['证券名称'] == stock_name]['交易市场'].iloc[0]
        market2 = df2[df2['证券名称'] == stock_name]['交易市场'].iloc[0]

        stock_num1 = df[df['证券名称'] == stock_name]['股票余额'].iloc[0]
        stock_num2 = df2[df2['证券名称'] == stock_name]['股票余额'].iloc[0]
        # tock_num = stock_num1 + stock_num2

        stock_cost1 = df[df['证券名称'] == stock_name]['成本价'].iloc[0]
        stock_cost2 = df2[df2['证券名称'] == stock_name]['成本价'].iloc[0]
        # tock_cost = stock_cost1*(stock_num1/stock_num) + stock_cost2*(stock_num2/stock_num)

        stock_price1 = df[df['证券名称'] == stock_name]['市价'].iloc[0]
        stock_price2 = df2[df2['证券名称'] == stock_name]['市价'].iloc[0]
        stock_price = (stock_price1 + stock_price2) / 2

        profit1 = df[df['证券名称'] == stock_name]['盈亏'].iloc[0]
        profit2 = df2[df2['证券名称'] == stock_name]['盈亏'].iloc[0]

        # 港股显示的交易市场是nan，汇率调整为rmb
        if 'nan' in str(market1):
            stock_cost1 = stock_cost1 * exchange_rate
            stock_price1 = stock_price1 * exchange_rate
            profit1 = profit1
        if 'nan' in str(market2):
            stock_cost2 = stock_cost2 * exchange_rate
            stock_price2 = stock_price2 * exchange_rate
            profit2 = profit2

        stock_num = stock_num1 + stock_num2  # 持股数
        profit = profit1 + profit2
        stock_cost = stock_cost1 * (stock_num1 / stock_num) + stock_cost2 * (stock_num2 / stock_num)  # 平均成本
        stock_price = stock_price1 * (stock_num1 / stock_num) + stock_price2 * (stock_num2 / stock_num)  # 平均成本
        market_cap = stock_num1 * stock_price1 + stock_num2 * stock_price2  # 总持仓市值
        profit_ratio = profit / (stock_cost1 * stock_num1 + stock_cost2 * stock_num2) * 100  # 盈亏比例

    else:
        market = df2[df2['证券名称'] == stock_name]['交易市场'].iloc[0]
        stock_num = df2[df2['证券名称'] == stock_name]['股票余额'].iloc[0]
        stock_cost = df2[df2['证券名称'] == stock_name]['成本价'].iloc[0]
        stock_price = df2[df2['证券名称'] == stock_name]['市价'].iloc[0]
        profit = df2[df2['证券名称'] == stock_name]['盈亏'].iloc[0]
        market_cap = stock_num * stock_price
        if 'nan' in str(market):
            stock_cost = stock_cost * exchange_rate
            stock_price = stock_price * exchange_rate
            profit = profit * exchange_rate
        profit_ratio = profit / (stock_cost * stock_num) * 100  # 盈亏比例（未扣费）

    print(stock_num, stock_cost, market_cap, profit_ratio)
    temp = temp.append(
        {'证券代码': stock_code, '证券名称': stock_name, '股票余额': stock_num, '成本价': stock_cost, '市价': stock_price, '盈亏': profit,
         '市值': market_cap,
         '盈亏比例': profit_ratio}, ignore_index=True)

# 处理df
for stock_name in df['证券名称']:
    stock_code = str(df[df['证券名称'] == stock_name]['证券代码'].iloc[0])
    print(stock_code)

    if stock_name not in df2['证券名称'].to_list():
        print(stock_name)
        market = df[df['证券名称'] == stock_name]['交易市场'].iloc[0]
        stock_num = df[df['证券名称'] == stock_name]['股票余额'].iloc[0]
        stock_cost = df[df['证券名称'] == stock_name]['成本价'].iloc[0]
        stock_price = df[df['证券名称'] == stock_name]['市价'].iloc[0]
        profit = df[df['证券名称'] == stock_name]['盈亏'].iloc[0]
        market_cap = stock_num * stock_price
        if 'nan' in str(market):
            stock_cost = stock_cost * exchange_rate
            stock_price = stock_price * exchange_rate
            profit = profit * exchange_rate

        profit_ratio = profit / (stock_cost * stock_num) * 100  # 盈亏比例
        print(stock_num, stock_cost, market_cap, profit_ratio)
        temp = temp.append(
            {'证券代码': stock_code, '证券名称': stock_name, '股票余额': stock_num, '成本价': stock_cost, '市价': stock_price,
             '盈亏': profit, '市值': market_cap,
             '盈亏比例': profit_ratio}, ignore_index=True)

# 计算仓位占比，提取行业
temp['仓位占比'] = temp['市值'] / total_asset * 100
temp['行业（申万二级）'] = 0
for i in range(len(temp)):
    stock_code = temp['证券代码'][i]
    print(stock_code)
    if int(stock_code) > 400000:
        code = str(stock_code) + '.XSHG'
    elif int(stock_code) < 400000:
        code = '0' * (6-len(str(stock_code))) + str(stock_code) + '.XSHE'
    try:
        industry = pd.DataFrame(myclient['stock_fundamentals']['industry_swl2'].find({'code': code, 'date': '2020-05-01'}))[
            'industry_sw_l2'][0]
        temp['行业（申万二级）'][i] = industry
    except Exception as e:
        print(str(e))

temp = temp.sort_values(by=["行业（申万二级）", "盈亏比例"], ascending=False)
temp = temp.set_index('证券代码')
temp['市值'] = temp['市值'].apply(lambda x: int(x))
temp['盈亏'] = temp['盈亏'].apply(lambda x: int(x))
temp['成本价'] = temp['成本价'].apply(lambda x: np.around(x, 2))
temp['盈亏比例'] = temp['盈亏比例'].apply(lambda x: np.around(x, 2))
temp['仓位占比'] = temp['仓位占比'].apply(lambda x: np.around(x, 2))



# 输出
temp[['证券名称', '股票余额', '成本价', '市价', '市值', '盈亏', '盈亏比例', '仓位占比', '行业（申万二级）']]\
    .to_excel(export_file_path)
