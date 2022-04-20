# -*- coding: utf-8 -*
# @Time    : 2021/12/9 14:16
# @Author  : gzy
# @File    : get_Rationale_distribution.py
import jsonlines
import matplotlib
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

def get_rationale_distribution():
    with jsonlines.open('../../data/para_scifact/') as f:
        pass

def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    file = open('lengths_dev.txt','r',encoding='utf-8')
    data = []
    for line in file:
        data.append(line.strip())
    #  matplotlib.axes.Axes.hist() 方法的接口
    # n, bins, patches = plt.hist(x=data, bins=10, color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('My Very Own Histogram')
    # plt.text(23, 45, r'$\mu=15, b=3$')
    # maxfreq = n.max()
    # # 设置y轴的上限
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    # plt.show()
    import seaborn as sns

    sns.set_style('darkgrid')
    kwargs = {'cumulative': True}
    sns.distplot(data,kde_kws=kwargs)
    plt.show()



if __name__ == '__main__':
    plot()
