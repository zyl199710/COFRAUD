
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
import scipy.sparse as sp
import numpy as np
from scipy.io import loadmat

import math

import matplotlib.pyplot as plt
import seaborn as sns


def computelocal():
    yelp = loadmat('data/Amazon.mat')

    A = yelp['net_upu']
    a = A.T > A
    A = A + np.multiply(A.T, a) - np.multiply(A, a)
    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set1 = set(row)
    G1 = nx.from_scipy_sparse_matrix(sp.coo_matrix(A))


    A = yelp['net_usu']
    a = A.T > A
    A = A + np.multiply(A.T, a) - np.multiply(A, a)
    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set2 = set(row)
    G2 = nx.from_scipy_sparse_matrix(sp.coo_matrix(A))


    A = yelp['net_uvu']
    a = A.T > A
    A = A + np.multiply(A.T, a) - np.multiply(A, a)
    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set3 = set(row)
    G3 = nx.from_scipy_sparse_matrix(sp.coo_matrix(A))

    set_all = set1 & set2 & set3
    labels = yelp['label'].flatten()

    cor1_2_t = []
    cor1_3_t = []
    cor2_3_t = []
    cor1_2_f = []
    cor1_3_f = []
    cor2_3_f = []

    for i in set_all:
        nei1 = list(G1.neighbors(i))
        nei2 = list(G2.neighbors(i))
        nei3 = list(G3.neighbors(i))
        if len(nei1) <2 or len(nei2)<2 or  len(nei3) < 2:
            continue
        j12 = adamic_adar_index(nei1, nei2)
        j13 = adamic_adar_index(nei1, nei3)
        j23 = adamic_adar_index(nei2, nei3)
        if labels[i] == 1:
            if j12 != 0:
                cor1_2_t.append(j12)
            if j13 != 0:
                cor1_3_t.append(j13)
            if j23 != 0:
                cor2_3_t.append(j23)
        if labels[i] == 0:
            if j12 != 0:
                cor1_2_f.append(j12)
            if j13 != 0:
                cor1_3_f.append(j13)
            if j23 != 0:
                cor2_3_f.append(j23)



    figure1 = plt.figure(1,figsize=(5.6,4.1))
    sns.distplot(cor1_2_t,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#7030A0",'linestyle':'--', "label": "Frausters"}, bins= 320 )
    sns.distplot(cor1_2_f,norm_hist= True, hist = True, kde= False, hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#D4B348",'linestyle':'--', "label": "Normal users"},bins=320) #sns.distplot(xx3,norm_hist= True, hist = True, kde_kws={"color": "#92D050", "lw": 3, "label": "Shortest Path Length = 3",'linestyle':'--'} , hist_kws={"histtype": "step", "linewidth": 1.5,"alpha": 0, "color": "#92D050"})
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Cross-relation neighbor overlap', fontsize = 18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=9)
    plt.xlim(-0.1, 4.1)
    #plt.ylim(0,0.4)
    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    #figure1.savefig('1-2.pdf', format='pdf')

    figure2 = plt.figure(2,figsize=(5.6,4.1))
    sns.distplot(cor1_3_t, norm_hist= True, hist = True,kde= False, hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#7030A0",'linestyle':'--', "label": "Frausters"}, bins= 30  )
    sns.distplot(cor1_3_f, norm_hist= True, hist = True,kde= False, hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#D4B348",'linestyle':'--', "label": "Normal users"},bins= 30)
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Cross-relation neighbor overlap', fontsize = 18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlim(-0.1, 4.1)
    #plt.ylim(0,5)
    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    #figure2.savefig('1-3.pdf', format='pdf')

    figure3 = plt.figure(3,figsize=(5.6,4.1))
    sns.distplot(cor2_3_t, norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#7030A0",'linestyle':'--', "label": "Frausters"}, bins=240 )
    sns.distplot(cor2_3_f, norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#D4B348",'linestyle':'--', "label": "Normal users"},bins=240)
    #sns.distplot(xx3,norm_hist= True, hist = True, kde_kws={"color": "#92D050", "lw": 3, "label": "Shortest Path Length = 3",'linestyle':'--'} , hist_kws={"histtype": "step", "linewidth": 1.5,"alpha": 0, "color": "#92D050"})
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Cross-relation neighbor overlap', fontsize = 18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlim(-0.1, 4.1)
    #plt.ylim(0,5)
    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    #figure3.savefig('2-3.pdf', format='pdf')
    plt.show()

def stander(x):
    x1 = (x - np.min(x))/(np.max(x) - np.min(x))
    return x1

def adamic_adar_index(x, y):
    x = set(x)
    y = set(y)
    sum = 0
    for i in x & y:
        sum = sum + 1/math.log(len(x)*len(y))
    return sum




computelocal()
#computeglobal()