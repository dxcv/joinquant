# -*- coding: utf-8 -*-
# @Time    : 2020/2/25 13:07
# @Author  : zhangfang
from __future__ import division
import pymongo
from arctic import Arctic, TICK_STORE, CHUNK_STORE
from jqdatasdk import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import math
import numpy as np
import json
import seaborn as sns

auth('18610039264', 'zg19491001')
style.use('ggplot')

def stock_price_jz(sec, sday, eday):
    """
    输入 股票代码，开始日期，截至日期
    输出 个股的后复权的开高低收价格
    """
    temp = jzmongo['stock_raw.wind_index'].read(sec)
    temp['trade_date'] = temp.index
    temp = temp.assign(trade_date=lambda df: df.trade_date.apply(lambda x: str(x)[:10])).dropna()
    if sec == '930606.CSI':
        print(temp)
    temp = temp.assign(high=lambda df: df.high.apply(lambda x: trans_heng_float(x)))\
        .assign(open=lambda df: df.open.apply(lambda x: trans_heng_float(x)))\
        .assign(low=lambda df: df.high.apply(lambda x: trans_heng_float(x)))[['high', 'open', 'low', 'close', 'trade_date']]
    temp = temp.fillna(method='backfill', axis=1)

    temp = temp[(temp['trade_date'] >= sday) & (temp['trade_date'] <= eday)]

    temp[['high', 'open', 'low', 'close']] = temp[['high', 'open', 'low', 'close']].astype(float)
    return temp


def trans_heng_float(x):
    if x == '--':
        x = None
    return x


if __name__ == "__main__":
    fold = 'e:/fof/ymjh/'
    fold_data = 'e:/fof/data/'
    myclient = pymongo.MongoClient('mongodb://juzheng:jz2018*@192.168.2.201:27017/')
    jzmongo = Arctic(myclient)
    end_day = datetime.date.today().strftime('%Y-%m-%d')
    trd_state_df = pd.read_csv(fold + 'state_ymjh_tb_11.csv', encoding='gbk', index_col=0)

    etf_to_idx = pd.read_excel(fold_data + 'etf_to_idx_ymjh_indus.xls', encoding='gbk')[
        ['基金代码', '基金简称', '跟踪指数', '跟踪指数代码', 'select']] \
        .rename(columns={'基金代码': 'code', '基金简称': 'name', '跟踪指数代码': 'idx_code', '跟踪指数': 'idx_name'})
    etf_to_idx = etf_to_idx[etf_to_idx['select'] == 1]
    etf_to_idx = etf_to_idx.drop_duplicates(['idx_code'], keep='first') \
        .assign(idx_code=lambda df: df.idx_code.apply(lambda x: str(x))).sort_values(['idx_code'])
    etf_to_idx = etf_to_idx.assign(code_trans=lambda df: df.code.apply(lambda x: str(x)[:6])) \
        .assign(idx_code=lambda df: df.idx_code.apply(lambda x: str(x)[:6]))
    jz_idx_code_lst = jzmongo['stock_raw.wind_index'].list_symbols()
    jz_idx_code_df = pd.DataFrame(jz_idx_code_lst, columns=['jz_code'])
    jz_idx_code_df['temp'] = jz_idx_code_df.jz_code.apply(lambda x: x[0])
    jz_idx_code_df = jz_idx_code_df[(jz_idx_code_df['temp'] == '9') | (jz_idx_code_df['temp'] == '0') |
                                    (jz_idx_code_df['temp'] == 'H') | (jz_idx_code_df['temp'] == '3')]
    jz_idx_code_df = jz_idx_code_df.assign(idx_code=lambda df: df.jz_code.apply(lambda x: x[:6]))
    jz_idx_code_df = jz_idx_code_df.merge(etf_to_idx, on=['idx_code'])
    # jz_idx_code_df.to_csv(fold_data + 'ymjh_select_idx_code.csv', encoding='gbk')

    trd_idx_code_list = jz_idx_code_df.jz_code.tolist()
    trd_idx_code_list = list(set(trd_idx_code_list))
    trd_idx_code_list.sort()
    print(trd_idx_code_list)

    s_period_lst = [i for i in range(40, 2, -2)]
    l_period_lst = [i for i in range(6, 83, 4)]
    # pecent_lst = [i / 100 for i in range(40, 96, 100)]

    name_lst = ['000932.SH', '000988.SH', '000991.SH', '399971.SZ', '399987.SZ', '399998.SZ', '930606.CSI',
                '930713.CSI', '980017.CNI', 'H30165.CSI', 'H30184.CSI']
    time_lst = [('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    para_dict = {'000932': [(4, 42), (21, 34), (10, 62), (24, 38), (10, 54)],
                 '000988': [(9, 48), (7, 16), (11, 34)],
                 '000991': [(7, 44), (7, 20), (5, 32)],
                 '399971': [(7, 20), (7, 20), (7, 36)],
                 '399987': [(27, 80), (12, 40), (27, 60)],
                 '399998': [(31, 70), (8, 16), (22, 64)],
                 '930606': [(7, 60), (7, 52), (8, 56), (12, 34)],
                 '930713': [(11, 28), (8, 14), (13, 20), (19, 64)],
                 '980017': [(9, 44), (7, 14), (21, 60), (19, 52), (7, 14)],
                 'H30165': [(5, 44), (8, 14), (8, 36), (8, 48)],
                 'H30184': [(9, 44), (7, 36), (17, 66), (6, 32), (22, 44)]
                 }
    # s_period_lst = [i for i in range(40, 22, -1)]
    # l_period_lst = [i for i in range(36, 76, 2)]
    name_lst = ['000932.SH']
    time_lst = [('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]

    # time_lst = [('2014-01-01', '2015-12-31')]

    for name, group in trd_state_df.groupby(['symble']):
        if name not in name_lst:
            continue
        for (s_date, e_date) in time_lst:
            group_ = group[(group['e_time'] <= e_date) & (group['s_time'] >= s_date)]
            if len(group_) < 10:
                continue
            group_ = group_.set_index(['period_s', 'period_l'])
            # trd_state_df_ = trd_state_df_[(trd_state_df_['s_date'] >= s_date) & (trd_state_df_['e_date'] <= e_date)]
            sharp_lst = []
            harvest = []
            for s_period in s_period_lst:
                harvest_row = []
                for l_period in l_period_lst:
                    if(s_period >= l_period):
                        sharp = 0
                        harvest_row.append(0)
                        continue
                    sharp = group_.loc[s_period, l_period].sharp
                    harvest_row.append(sharp)
                    print(sharp)
                    sharp_lst.append(sharp)
                harvest.append(harvest_row)
            x_label = s_period_lst
            y_label = l_period_lst
            # print(harvest)
            harvest = np.array(harvest)
            fig, ax1 = plt.subplots(figsize=(len(x_label), len(y_label)), nrows=1)

            vmax = max(max(harvest[i]) for i in range(len(harvest)))
            vmin = -vmax
            print(vmax)
            if vmax < 0:
                vmin = min(min(harvest[i]) for i in range(len(harvest)))

            h = sns.heatmap(harvest, annot=True, fmt='.2f', ax=ax1, vmax=vmax, vmin=vmin, annot_kws={'size': 20},
                            cbar=False)
            cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
            cb.ax.tick_params(labelsize=28)
            ax1.set_title(s_date + '_' + name, fontsize=32)
            ax1.set_xticklabels(l_period_lst, fontsize=20)
            ax1.set_yticklabels(s_period_lst, fontsize=20)
            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

            ax1.set_xlabel('l_period', fontsize=24)
            ax1.set_ylabel('s_period', fontsize=24)
            fig.tight_layout()
            plt.savefig(fold + 'fig/' + s_date + '_' + name + '.png')
            plt.show()