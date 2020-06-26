# GraphSTONE
A TensorFlow implementation of GraphSTONE, as described in our paper: 

>  Graph Structural-topic Neural Network (KDD 2020, Research Track)

http://arxiv.org/abs/2006.14278

#### Run
`python main.py`

#### Parameters

For parameter settings, please see `conf.json`



| Name  |  Default| Note|

| :-------- | --------:| :--: |

| Computer | 1600 元 | 5 |

| Phone  | 12 元 | 12 |

| Pipe   |  1 元 | 234 |



#### Evaluation
We provide node classification and link prediction tasks in `utils.py`.

#### Data
We provide cora and ppi datasets as examples under `data/cora` and `data/ppi`. 

Note that the `data/dataset_name/features.npy` has undergone a dimensionality reduction via PCA, and is not identical to the original cora features.

#### Acknowledgments

Certain parts of this project are partially derived from [GraLSP](https://github.com/KL4805/GRALSP) and [AnchorRecovery](https://github.com/CatalinVoss/anchor-baggage/tree/master/anchor-word-recovery)

