# GraphSTONE
A TensorFlow implementation of GraphSTONE, as described in our paper: 

>  Graph Structural-topic Neural Network (KDD 2020, Research Track)

http://arxiv.org/abs/2006.14278

#### How to Use
`python main.py`

#### Dependencies

Tensorflow 1.10.0
Networkx 1.11
Python 3

#### Data
We provide cora and ppi datasets as examples under `data/cora` and `data/ppi`. 

Note that the `data/dataset_name/features.npy` has undergone a dimensionality reduction via PCA, and is not identical to the original cora features.

#### Parameters
For parameter settings, please see `conf.json`.

Some parameter definitions:

| Name                    | Default |                Note                 |
| :---------------------- | ------: | :---------------------------------: |
| dataset                 |    cora |            dataset name             |
| input_node_feature      |    True | input original node features or not |
| PreProcess/number_paths |  50 |                 number of paths from a center node, for generating "word" and "document" concepts on graphs                 |
| PreProcess/path_length |  15 |                 max length of random walks from a center node, for generating "word" and "document" concepts on graphs                 |
| TopicModel/number_topic |   5 |                 number of structural-topics                 |
| TopicModel/max_features_dim |  2500 |                 max topic_features (for the input of structural-topic GNN) dimension                 |
| TopicGCN/max_training_steps         | 5000 |            max steps for training            |


#### Acknowledgments

Certain parts of this project are partially derived from [GraLSP](https://github.com/KL4805/GRALSP) and [AnchorRecovery](https://github.com/CatalinVoss/anchor-baggage/tree/master/anchor-word-recovery).

