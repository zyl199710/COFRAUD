#import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import numpy as np
from scipy.io import loadmat

def computeglobal():
    yelp = loadmat('data/Amazon.mat')
    import networkx as nx
    A = yelp['net_upu']
    a = A.T > A
    A1 = A + np.multiply(A.T, a) - np.multiply(A, a)

    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set1 = set(row)


    A = yelp['net_usu']
    a = A.T > A
    A2 = A + np.multiply(A.T, a) - np.multiply(A, a)
    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set2 = set(row)

    A = yelp['net_uvu']
    a = A.T > A
    A3 = A + np.multiply(A.T, a) - np.multiply(A, a)
    tmp_coo = sp.coo_matrix(A)
    row = tmp_coo.row
    set3 = set(row)

    #a= A1@A2
    A1 = sp.coo_matrix(A1)
    A2 = sp.coo_matrix(A2)
    A3 = sp.coo_matrix(A3)

    # a = A1*A2
    # A12 = A1+A2-a
    #
    # a = A1*A3
    # A13 = A1+A3-a
    #
    # a = A2*A3
    # A23 = A2+A3-a

    #A12 =

    G1 = nx.from_scipy_sparse_matrix(A1)
    pr1 = nx.pagerank(G1)
    #np.save('pagerank1.npy', pr1)
    G2 = nx.from_scipy_sparse_matrix(A2)
    pr2 = nx.pagerank(G2)
    #np.save('pagerank13.npy', pr13)

    G3 = nx.from_scipy_sparse_matrix(A3)
    pr3 = nx.pagerank(G3)
    #np.save('pagerank23.npy', pr3)

    # pr1 = np.load('pagerank1.npy', allow_pickle=True)
    # pr1 = pr1.item()
    # pr2 = np.load('pagerank2.npy', allow_pickle=True)
    # pr2 = pr2.item()
    # pr3 = np.load('pagerank3.npy', allow_pickle=True)
    # pr3 = pr3.item()


    set_all = set1 & set2 & set3
    labels = yelp['label'].flatten()
    prr1_t = []
    prr2_t = []
    prr3_t = []
    prr1_f = []
    prr2_f = []
    prr3_f = []
    cor1_2_t = []
    cor1_3_t = []
    cor2_3_t = []
    cor1_2_f = []
    cor1_3_f = []
    cor2_3_f = []

    G1 = nx.from_scipy_sparse_matrix(A1)
    G2 = nx.from_scipy_sparse_matrix(A2)
    G3 = nx.from_scipy_sparse_matrix(A3)

    for i in set_all:
        nei1 = list(G1.neighbors(i))
        nei2 = list(G2.neighbors(i))
        nei3 = list(G3.neighbors(i))
        if len(nei1) <2 or len(nei2)<2 or  len(nei3) < 2:
            continue
        if labels[i] == 1:
            prr1_t.append(pr1[i] * 10000)
            prr2_t.append(pr2[i] * 10000)
            prr3_t.append(pr3[i] * 10000)
            cor1_2_t.append(abs(pr1[i] - pr2[i]))
            cor1_3_t.append(abs(pr1[i] - pr3[i]))
            cor2_3_t.append(abs(pr3[i] - pr2[i]))
        if labels[i] == 0:
            prr1_f.append(pr1[i] * 10000)
            prr2_f.append(pr2[i] * 10000)
            prr3_f.append(pr3[i] * 10000)
            cor1_2_f.append(abs(pr1[i] - pr2[i]))
            cor1_3_f.append(abs(pr1[i] - pr3[i]))
            cor2_3_f.append(abs(pr3[i] - pr2[i]))




    figure1 = plt.figure(1,figsize=(5.6,4.1))
    plt.style.use('classic')
    sns.distplot(prr1_t,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#f47920",'linestyle':'--', "label": "Frausters"}, bins= 80 )
    sns.distplot(prr1_f,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#669bcd",'linestyle':'--', "label": "Normal users"},bins=80) #sns.distplot(xx3,norm_hist= True, hist = True, kde_kws={"color": "#92D050", "lw": 3, "label": "Shortest Path Length = 3",'linestyle':'--'} , hist_kws={"histtype": "step", "linewidth": 1.5,"alpha": 0, "color": "#92D050"})
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Pagerank ($\\times 10^{-4}$)', fontsize = 18)
    plt.ylabel('Density ($\\times 10^{4}$)', fontsize=18)
    plt.ticklabel_format(style='sci', axis='both')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=11)
    plt.xlim(-0.1, 3.1)
    #plt.ylim(0,0.4)

    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    figure1.savefig('pr1.pdf', format='pdf')

    figure2 = plt.figure(2,figsize=(5.6,4.1))
    plt.style.use('classic')
    sns.distplot(prr2_t,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#f47920",'linestyle':'--', "label": "Frausters"}, bins= 45 )
    sns.distplot(prr2_f,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#669bcd",'linestyle':'--', "label": "Normal users"},bins=45) #sns.distplot(xx3,norm_hist= True, hist = True, kde_kws={"color": "#92D050", "lw": 3, "label": "Shortest Path Length = 3",'linestyle':'--'} , hist_kws={"histtype": "step", "linewidth": 1.5,"alpha": 0, "color": "#92D050"})
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Pagerank ($\\times 10^{-4}$)', fontsize = 18)
    plt.ylabel('Density ($\\times 10^{4}$)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=11)
    plt.xlim(-0.1, 3.1)
    #plt.ylim(0,0.4)
    plt.ticklabel_format(style='sci', axis= 'both')
    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    figure2.savefig('pr2.pdf', format='pdf')

    figure3 = plt.figure(3 ,figsize=(5.6,4.1))
    plt.style.use('classic')
    sns.distplot(prr3_t,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#f47920",'linestyle':'--', "label": "Frausters"}, bins= 100 )
    sns.distplot(prr3_f,norm_hist= True, hist = True, kde= False,  hist_kws={"histtype": "step", "linewidth": 3,"alpha": 0.8, "color": "#669bcd",'linestyle':'--', "label": "Normal users"},bins=100) #sns.distplot(xx3,norm_hist= True, hist = True, kde_kws={"color": "#92D050", "lw": 3, "label": "Shortest Path Length = 3",'linestyle':'--'} , hist_kws={"histtype": "step", "linewidth": 1.5,"alpha": 0, "color": "#92D050"})
    plt.legend(loc='upper right',fontsize = 18)
    plt.xlabel('Pagerank ($\\times 10^{-4}$)', fontsize = 18)
    plt.ylabel('Density ($\\times 10^{4}$)', fontsize =  18)
    plt.ticklabel_format(style = "sci", axis= "both")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlim(-0.1, 3.1)


    #plt.ylim(0,0.4)
    plt.subplots_adjust(left=0.13, bottom=0.17, right=0.943, top=0.936)
    figure3.savefig('pr3.pdf', format='pdf')
    plt.show()

computeglobal()