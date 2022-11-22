import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plt_save(f_name):
    plt.savefig(f_name, dpi=480, bbox_inches='tight')

def plot_fig1():
    # size - F1
    lf_lstm = plt.scatter(x=1.24, y=77.8, c='blue', marker='P')     # LF-LSTM
    mult = plt.scatter(x=1.07, y=80.4, c='firebrick', marker='d')        # MulT
    lmf_mult = plt.scatter(x=0.84, y=78.5, c='slateblue', marker='s')    # LMF-MulT
    pmr = plt.scatter(x=2.14, y=82.1, c='royalblue', marker='*')         # PMR
    lmr_cbt = plt.scatter(x=0.34, y=81.0, c='teal', marker='o')      # LMR-CBT
    hcmt = plt.scatter(x=0.92, y=82.6, c='red', marker='^')        # HCMT

    models = [lf_lstm, mult, lmf_mult, pmr, lmr_cbt, hcmt]
    labels = ['LF-LSTM', 'MulT', 'LMF-MulT', 'PMR', 'LMR-CBT', 'HCMT(ours)']
    plt.legend(handles=models, labels=labels, loc='lower right')
    plt.grid()

    plt.ylabel('F1-Score(%)')
    plt.xlabel('Parameters(M)')
    plt_save("./figs/fig_1.svg")
    plt.show()

def plot_fig3():
    # plt.style.use('classic')
    attention = np.array(
        [[0.5, 0.4, 0.1],
         [0.2, 0.5, 0.3],
         [0.2, 0.4, 0.4],
         [0.9, 1.3, 0.8]
         ]
    )
    a_sum = np.array([[0.9, 1.3, 0.8]])

    heatmap = plt.matshow(attention)
    for (i, j), z in np.ndenumerate(attention):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
                # bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.1'))
    plt.xlabel("Source Modal", labelpad=-270)
    plt.ylabel("Target Modal")

    y_ = ['I', 'Eat', 'Pizza', '(sum)']
    x_ = ['I', 'Eat', 'Pizza']
    plt.xticks([0, 1, 2], x_)
    plt.yticks([0, 1, 2, 3], y_)

    plt.colorbar(heatmap)
    plt_save('./figs/fig_3.svg')
    plt.show()

    ...

# plot_fig3()
plot_fig1()
