# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 16:00:44 2018

@author: Administrator
"""

from jqdatasdk import *
import datetime

auth('18610039264', 'zg19491001')

"""
1、以下函数中 股票代码的格式为 '000001.XSHE','603980.XSHG'，日期格式为 '2018-01-01'
2、开通权限后，您可以在本地Python环境下安装和使用JQData，具体操作方法如下。如使用中遇到问题，可以添加JQData管理员微信，管理员微信号：jqdata01，添加时请留言"JQData"。

安装JQData：打开本地cmd终端或Mac终端，将路径切换到python目录下，使用pip语法输入如下语句 ，安装JQData数据包：

pip install git+https://github.com/JoinQuant/jqdatasdk.git -i https://mirrors.aliyun.com/pypi/simple/
如果上述方法安装失败，请使用备选安装方法：下载JQData压缩包，解压压缩包，打开终端并切换到jqdatasdk-master文件中setup.py所在的目录下，执行 python setup.py install，也能完成安装。

升级JQData：JQData预计每2周会发布一次迭代版本，增加更多维度的基础数据以及因子类数据，感兴趣的用户可以查看JQData的版本信息，查看新增的数据种类。

pip install -U git+https://github.com/JoinQuant/jqdatasdk.git
登录JQData：安装成功后，导入jqdatasdk，并认证用户身份，认证的账号和密码是用户在聚宽官网注册的账号和密码（只有开通jqdatasdk使用权限的用户才能成功认证）。认证成功后，显示"auth success"，就可以开始使用数据了，认证代码如下。

from jqdatasdk import *
auth('聚宽账号','登录密码')
3、开通权限的链接  https://www.joinquant.com/default/index/sdk#jq-sdk-apply  
4、接口说明文档   https://www.joinquant.com/help/api/help?name=JQData#聚宽JQData用户使用说明
"""


# 获取当前全部股票信息
def all_stocks(date):
    """
    types: list: 用来过滤securities的类型, list元素可选: 'stock', 'fund', 'index', 'futures', 'options', 'etf', 'lof', 'fja',
     'fjb', 'open_fund', 'bond_fund', 'stock_fund', 'QDII_fund', 'money_market_fund', 'mixture_fund'。types为空时返回所有股票,
     不包括基金,指数和期货
    date: 日期, 一个字符串或者 [datetime.datetime]/[datetime.date] 对象, 用于获取某日期还在上市的股票信息. 默认值为 None, 表示获取所有日期的股票信息
    """
    code_lst = get_all_securities(types=['stock'], date=date).index.tolist()
    code_lst = [i for i in code_lst if i[1] != '8']
    return code_lst


# 获取当前各大指数的成分股
def index_stocks(_index):
    """
    输入 指数编码：000016.XSHG	上证50；000300.XSHG	沪深300；399005.XSHE	中小板指
                 399006.XSHE	创业板指；000905.XSHG	中证500
    返回 成分股代码列表
    输出格式 list
    """
    return get_index_stocks(_index)


# 对各种格式的代码标准化
def code_chg_format(code):
    """
    输入 各种形式的股票代码
    输出 标准格式的股票代码
    """
    return normalize_code(code)


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


# 获取交易日
def tradeday(sday, eday):
    """
    输入 开始时间 和 截止时间
    输出 list 交易日 datetime格式
    """
    return get_trade_days(sday, eday)


# 获取个股行情
def stock_price(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = get_price(sec, start_date=sday, end_date=eday, frequency='daily', fields=None, skip_paused=False, fq='post',
                     count=None).reset_index() \
        .rename(columns={'index': 'tradedate', 'date': 'date_time'})
    temp['stock_code'] = sec
    return temp


# 获取个股行情
def get_stock_price(sec, count, unit, include_now, end_dt, fq_ref_date):
    """
    security: 股票代码，支持单个及多个标的
    count: 大于0的整数，表示获取bar的个数。如果行情数据的bar不足count个，返回的长度则小于count个数。
    unit: bar的时间单位, 支持如下周期：'1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'。其中m表示分钟，d表示天，w表示周，M表示月。
    fields: 获取数据的字段， 支持如下值：'date', 'open', 'close', 'high', 'low', 'volume', 'money', 'open_interest'。
    include_now: 取值True 或者False。 表示是否包含当前bar, 比如策略时间是9:33，unit参数为5m， 如果 include_now=True,则返回9:30-9:33这个分钟 bar。
    end_dt：查询的截止时间，支持的类型为datetime.datetime或None，默认为datetime.now()。
    fq_ref_date：复权基准日期，为None时为不复权数据。

    """
    temp = get_bars(sec, count, unit=unit, fields=['date', 'open', 'high', 'low', 'close', 'volume', 'money'],
                    include_now=include_now, end_dt=end_dt, fq_ref_date=fq_ref_date) \
        .assign(stock_code=sec)\
        .rename(columns={'date': 'date_time'})
    return temp


# 获取市值表，反映每日变化的市值，换手率以及估值的相关信息
def values_data(stockcode, eday, counts):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe 市值表 字段为 code :股票代码  day:日期  capitalization:总股本（万股）
    circulating_cap ：流通股本（万股） market_cap：总市值（亿） circulating_market_cap：流通市值（亿）
    turnover_ratio：换手率 pe_ratio：市盈率 TTM pe_ratio_lyr：静态市盈率  pb_ratio：市净率
    ps_ratio：市销率  pcf_ratio：市现率
    """
    q = query(valuation
              ).filter(valuation.code == stockcode)

    panel = get_fundamentals_continuously(q, end_date=eday, count=counts)
    return panel.minor_xs(stockcode)


# 获取资产负债表，反映财务三大表中资产负债表的相关字段
def balance_data(stockcode, eday, counts):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe https://www.joinquant.com/help/api/help?name=Stock#资产负债数据
    """
    q = query(balance
              ).filter(balance.code == stockcode)

    panel = get_fundamentals_continuously(q, end_date=eday, count=counts)
    return panel.minor_xs(stockcode)


# 获现金流量表，反映财务三大表中现金流量表的相关字段
def cash_flow_data(stockcode, eday, counts):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe https://www.joinquant.com/help/api/help?name=Stock#现金流数据
    """
    q = query(cash_flow
              ).filter(cash_flow.code == stockcode)

    panel = get_fundamentals_continuously(q, end_date=eday, count=counts)
    return panel.minor_xs(stockcode)


# 获利润表，反映财务三大表中利润表的相关字段
def income_data(stockcode, eday, counts):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe https://www.joinquant.com/help/api/help?name=Stock#利润数据
    """
    q = query(income
              ).filter(income.code == stockcode)

    panel = get_fundamentals_continuously(q, end_date=eday, count=counts)
    return panel.minor_xs(stockcode)


# 获财务分析表，反映财务三大表中财务分析的相关字段
def indicator_data(stockcode, eday, counts):
    """
    输入 股票代码，查询中止日，以及查询多少条数据
    输出 dataframe https://www.joinquant.com/help/api/help?name=Stock#财务指标数据
    """
    q = query(indicator
              ).filter(indicator.code == stockcode)

    panel = get_fundamentals_continuously(q, end_date=eday, count=counts)
    return panel.minor_xs(stockcode)


def get_all_stock_name():
    data = get_all_securities(types=['stock'], date=None).reset_index(drop=False)[['index', 'display_name']]\
        .rename(columns={'index': 'stock_code', 'display_name': 'stock_name'})
    return data


if __name__ == "__main__":
    today = datetime.date.today()
    print(today)
    code_lst300 = get_index_stocks('000300.XSHG', '2019-09-22')
    print(code_lst300)
