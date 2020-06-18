#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import networkx as nx
import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


dataset = "cora"
embedding_file = os.path.join("./data/output", "dataset", "embeddings.npy")

print(embedding_file)

edge_file = os.path.join("./data/output", "dataset", "edges.txt")
flag_file = os.path.join("./data/output", "dataset", "labels.txt")


def sample_edges(G, ratio):
    # sample positive and neg edges
    pos_edges, neg_edges = [], []

    num_nodes = G.number_of_nodes()

    for _ in range(int(ratio * num_nodes)):
        u, v = random.choice(list(G.edges()))
        G.remove_edge(u, v)
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


def link_prediction(G, ratio, embedding_file):
    pos_edges, neg_edges = sample_edges(G, ratio)
    embeddings = np.load(embedding_file)

    num_pos = len(pos_edges)

    scores = []
    for u, v in pos_edges:
        score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
            np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
        # score = np.linalg.norm(embeddings[u] - embeddings[v])
        scores.append(score)
    for u, v in neg_edges:
        score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
            np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
        # score = np.linalg.norm(embeddings[u] - embeddings[v])
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
    print(auc, recall)

    return auc, recall


def node_class(X, y, train_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rate, random_state=random.randint(0, 100))
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    y_tst_pred = clf.predict(X_test)
    return  metrics.precision_score(y_test, y_tst_pred, average='micro'), \
           metrics.f1_score(y_test, y_tst_pred, average='macro')

def multi_label_node_class(X, y, train_rate):
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=train_rate, random_state=random.randint(0, 100))
    clf = LogisticRegression(solver='liblinear', max_iter=800, multi_class='ovr')
    clf = OneVsRestClassifier(clf)
    clf = clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)

    macro_f1 = metrics.f1_score(test_y, predict_y, average='macro')
    micro_f1 = metrics.f1_score(test_y, predict_y, average='micro')
    return (micro_f1, macro_f1)


if __name__ == '__main__':
    G = nx.read_edgelist(edge_file, create_using=nx.Graph())

    # link prediction
    ratio = 0.3
    aucs, recalls = [], []
    for _ in range(5):
        auc, recall = link_prediction(G, ratio, embedding_file)
        aucs.append(auc)
        recalls.append(recall)

    print("avg auc: ", round(np.mean(aucs) * 100, 2))
    print("avg recall: ", round(np.mean(recalls) * 100, 2))



    # node classification
    X = np.array(np.load(embedding_file))

    y_dict = dict()
    for line in open(flag_file):
        line = line.strip().strip('\n').split('\t')
        y_dict[int(line[0])] = int(line[1])

    y = list()
    for i in range(G.number_of_nodes()):
        y.append(y_dict[i])
    y = np.array(y)

    train_rate = [0.3, 0.7]
    for t in train_rate:
        acc, macf1 = [], []
        for i in range(10):
            t1, t2 = eval(X, y, t)
            acc.append(t1)
            macf1.append(t2)

        print ("node classification: train_rate: ", t, "acc: ", np.var(acc)*100)
        print ("node classification: train_rate: ", t, "macf1: ", np.var(macf1)*100)