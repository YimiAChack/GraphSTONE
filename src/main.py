#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import tensorflow as tf
import utils 
from preprocess import PreProcess
from topic_model import GraphAnchorLDA
from topic_gcn import StructuralTopicGCN


if __name__ == "__main__":
    params = utils.load_json_file("../config.json")

    preprocesser = PreProcess(params)
    preprocesser.generate_topic_concepts()

    graph_topic_model = GraphAnchorLDA(params)
    graph_topic_model.learn_topic_distribution()
    graph_topic_model.generate_topic_features()

    topic_gcn_model = StructuralTopicGCN(params)
    topic_gcn_model.train()


    # evaluate 
    evaluator = utils.Evaluate(params)
    if params["dataset"] != "ppi":
        evaluator.ten_times_node_classification(flag_multi_label = False)
    else:
        # multi_label node classification
        evaluator.ten_times_node_classification(flag_multi_label = True)

    # evaluator.five_times_link_prediction()
