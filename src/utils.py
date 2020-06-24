#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import networkx as nx
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s', "", s)

    params = json.loads(s)

    batch_size = params["TopicGCN"]["batch_size"]
    num_skips = params["TopicGCN"]["num_skips"]
    assert batch_size % num_skips == 0

    path_length = params["TopicGCN"]["path_length"]
    window_size = params["TopicGCN"]["window_size"]
    assert path_length >= window_size 

    print("dataset:", params["dataset"])
    return params


class Evaluate(object):
    """ generate topic concepts, for the input of GraphAnchorLDA
    Args:
        path_edges: number of random walks started from a center node
        task: length of random walks
    """
    def __init__(self, params): 
        dataset = params["dataset"]
        self.embedding_file = os.path.join("../data/output", dataset, "model", "final_embeddings.npy")
        self.edge_file = os.path.join("../data/input", dataset, "edges.txt")
        self.flag_file = os.path.join("../data/input", dataset, "labels.txt")
        self.G = nx.read_edgelist(self.edge_file, create_using=nx.Graph())


    def sample_edges(self, ratio):
        ''' sample positive and neg edges
        '''
        pos_edges, neg_edges = [], []

        num_nodes = self.G.number_of_nodes()

        for _ in range(int(ratio * num_nodes)):
            u, v = random.choice(list(self.G.edges()))
            self.G.remove_edge(u, v)
            pos_edges.append((u, v))

        cnt = 0
        edge_list = list(G.edges())
        while cnt <= ratio * num_nodes:
            u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if (u, v) in edge_list or (v, u) in edge_list or (u, v) in pos_edges or (v, u) in pos_edges or (
                    u, v) in neg_edges or (v, u) in neg_edges:
                continue
            neg_edges.append((u, v))
            cnt += 1

        return pos_edges, neg_edges


    def link_prediction(self, ratio):
        pos_edges, neg_edges = self.sample_edges(self.G, ratio)
        embeddings = np.load(self.embedding_file)

        num_pos = len(pos_edges)

        scores = []
        for u, v in pos_edges:
            score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
                np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
            scores.append(score)
        for u, v in neg_edges:
            score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
                np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
            scores.append(score)

        argsorted = np.argsort(scores)

        num = len(argsorted)
        label = []
        for _ in range(num // 2):
            label.append(1)
        for _ in range(num - num // 2):
            label.append(0)
        label = np.array(label)

        pred = np.zeros(num)
        for i in range(num // 2 - 1, num):
            pred[argsorted[i]] = 1

        recall = metrics.recall_score(label, pred)
        fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return auc, recall


    def node_classification(self, X, y, train_rate):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rate, random_state=random.randint(0, 100))
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train, y_train)
        y_tst_pred = clf.predict(X_test)
        return  metrics.precision_score(y_test, y_tst_pred, average='micro'), \
               metrics.f1_score(y_test, y_tst_pred, average='macro')


    def multi_label_node_classification(self, X, y, train_rate):
        train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=train_rate, random_state=random.randint(0, 100))
        clf = LogisticRegression(solver='liblinear', max_iter=800, multi_class='ovr')
        clf = OneVsRestClassifier(clf)
        clf = clf.fit(train_x, train_y)
        predict_y = clf.predict(test_x)

        macro_f1 = metrics.f1_score(test_y, predict_y, average='macro')
        micro_f1 = metrics.f1_score(test_y, predict_y, average='micro')
        return (micro_f1, macro_f1)


    def five_times_link_prediction(self, ratio = 0.3):
        aucs, recalls = [], []
        for _ in range(5):
            auc, recall = self.link_prediction(ratio)
            aucs.append(auc)
            recalls.append(recall)

        print("avg auc of link prediction: ", round(np.mean(aucs) * 100, 2))
        print("avg recall of link prediction: ", round(np.mean(recalls) * 100, 2))


    def ten_times_node_classification(self, flag_multi_label):
        X = np.array(np.load(self.embedding_file))

        if flag_multi_label:
            y = np.loadtxt(self.flag_file)
        else:
            y_dict = dict()
            for line in open(self.flag_file):
                line = line.strip().strip('\n').split(' ')
                y_dict[int(line[0])] = int(line[1])

            y = list()
            for i in range(self.G.number_of_nodes()):
                y.append(y_dict[i])
            y = np.array(y)


        train_rate = [0.3, 0.7]
        for t in train_rate:
            acc, macf1 = [], []
            for i in range(10):
                if flag_multi_label:
                    t1, t2 = self.multi_label_node_classification(X, y, t)
                else:
                    t1, t2 = self.node_classification(X, y, t)
                acc.append(t1)
                macf1.append(t2)

            print ("10 times node classification: train_rate:", t, "macf1:", round(np.mean(macf1) * 100, 2))
            print ("10 times node classification: train_rate:", t, "acc:", round(np.mean(acc) * 100, 2))
