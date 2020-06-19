#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import tensorflow as tf
import utils 
from preprocess import Walks
from topic_model import GraphAnchorLDA
from topic_gcn import StructuralTopicGCN


if __name__ == "__main__":
    params = utils.load_json_file("../config.json")
    
    walk = Walks(params)
    walk.generate_topic_concepts()

    graph_topic_model = GraphAnchorLDA(params)
    graph_topic_model.learn_topic_distribution()
    graph_topic_model.select_topic_features()

    topic_gcn_model = StructuralTopicGCN(params)
    topic_gcn_model.train()

