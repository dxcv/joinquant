# -*- coding: utf-8 -*-
# @Time    : 2020/2/25 13:07
# @Author  : zhangfang

import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import json
import seaborn as sns

if __name__ == "__main__":
    fold = 'e:/fof/ymjh/'
    trd_state_df = pd.read_csv(fold + 'state_ymjh_tb_3.csv', encoding='gbk', index_col=0)


    s_period_lst = [i for i in range(40, 2, -2)]
    l_period_lst = [i for i in range(6, 83, 4)]
    # pecent_lst = [i / 100 for i in range(40, 96, 100)]

    name_lst = ['000068.XSHG', '399364.XSHE', '000932.XSHG', '000989.XSHG', '399973.XSHE', '000928.XSHG', '000066.XSHG',
                '399550.XSHE', '399967.XSHE', '000063.XSHG', '000036.XSHG', '000048.XSHG', '000986.XSHG', '000987.XSHG',
                '000814.XSHG', '000056.XSHG']
    time_lst = [('2008-01-01', '2009-12-31'), ('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    para_dict = {'hs300': [(17, 70), (7, 26), (6, 46), (5, 48), (9, 76)],
                 'sz50': [(22, 66), (7, 28), (6, 62), (8, 62), (10, 82)],
                 'zz500': [(5, 38), (21, 32), (8, 52), (20, 30), (6, 34)],
                 'cyb': [(4, 26), (29, 46), (6, 28), (18, 30), (35, 48)]}
    para_dict = {'INX': [(10, 70), (26, 66), (24, 58), (15, 32), (26, 46), (10, 18)],
                 'IXIC': [(9, 40), (14, 22), (10, 38), (14, 30), (6, 26), (20, 34)],
                 'DJI': [(10, 74), (32, 66), (26, 40), (26, 62), (26, 38), (30, 46)]}
    # s_period_lst = [i for i in range(40, 22, -1)]
    # l_period_lst = [i for i in range(36, 76, 2)]
    name_lst = ['000300.XSHG', '000016.XSHG', '000905.XSHG', '399006.XSHE']
    name_lst = ['INX', 'IXIC', 'DJI']
    name_lst = ['DJI']

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