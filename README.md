# Don't Ignore Alienation and Marginalization: Correlating Fraud Detection

The open-resourced implementation for IJCAI2023 Submission - COFRAUD.

COFRAUD is a correlation-aware fraud detection model, which innovatively incorporates synergistic camouflage into fraud detection. It achieves significant improvements over state-of-the-art methods.

## Environments

COFRAUD framework is implemented on Google Colab and major libraries include:

- [Pytorch = 2.1.0 + cu121](https://pytorch.org/)

- [Networkx](https://networkx.org/)

- [dgl = 2.0.0 + cu121](https://www.dgl.ai/)

- [numpy](https://github.com/numpy/numpy)


## Datasets

All dataset in this paper are from previous works

- Amazon[<sup>1</sup>](#refer-anchor-1) contains 11944 users (9.5% fraudsters) and three types of relation, U-P-U, U-S-U, and U-V-U.
  
- Yelp[<sup>2</sup>](#refer-anchor-2) contains 45954 users (14.5% fraudsters) and three types of relation, R-U-R, R-T-R, and R-S-R.


## Preliminary

In this paper, we design two statistics to prove the existence of alienation and marginalization of fraudsters.

Please Run:

``` python preprocess_data/alienation.py  ```

``` python preprocess_data/marginalization.py```  

Please see the example <font color="orange">COFRAUD.ipynb</font>, it is run on Google Colab.

## Method

Please see the example in <font color="orange">COFRAUD.ipynb</font>, it is run on Google Colab.


## Baseline

- [GraphConsis](https://github.com/safe-graph/DGFraud)

- [CARE-GNN](https://github.com/YingtongDou/CARE-GNN)

- [PC-GNN](https://github.com/PonderLY/PC-GNN)

- [FRAUDRE](https://github.com/FraudDetection/FRAUDRE)

- [BWGNN](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)





## Reference

<div id="refer-anchor-1"></div>

- [1] McAuley J J, Leskovec J. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews[C]//Proceedings of the 22nd international conference on World Wide Web. 2013: 897-908.

<div id="refer-anchor-2"></div>

- [2] Rayana S, Akoglu L. Collective opinion spam detection: Bridging review networks and metadata[C]//Proceedings of the 21th acm sigkdd international conference on knowledge discovery and data mining. 2015: 985-994.
