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
    fold = 'e:/fof/cgo/'
    trd_state_df = pd.read_csv(fold + 'trd_state_df2.csv', encoding='gbk', index_col=0)

    s_period_lst = [i for i in range(29, 2, -2)]
    l_period_lst = [i for i in range(6, 60, 4)]
    s_period_lst = [i for i in range(20, 4, -1)]
    l_period_lst = [i for i in range(10, 41, 2)]
    pecent_lst = [i / 100 for i in range(40, 96, 100)]

    name_lst = ['cyb', 'hs300', 'zz500', 'sz50']
    time_lst = [('2010-01-01', '2011-12-31'), ('2012-01-01', '2013-12-31'), ('2014-01-01', '2015-12-31'),
                ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31')]
    # s_period_lst = [i for i in range(23, 3, -1)]
    # l_period_lst = [i for i in range(10, 41, 2)]
    # pecent_lst = [i / 100 for i in range(65, 96, 5)]
    # name_lst = ['sz50']
    # time_lst = [('2018-01-01', '2015-12-31')]
    for (s_date, e_date) in time_lst:
        for pecent in pecent_lst:
            for name in name_lst:
                group_ = trd_state_df[(trd_state_df['pecent'] == pecent) & (trd_state_df['indus_name'] == name) &
                                      (trd_state_df['s_date'] == s_date)].set_index(
                    ['s_period', 'l_period'])
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
                ax1.set_title(s_date + '_' + str(int(100*pecent)) + '_' + name, fontsize=32)
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
                plt.savefig(fold + 'fig/' + s_date + '_' + str(int(100*pecent)) + '_' + name + '.png')
                plt.show()