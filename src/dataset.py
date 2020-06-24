#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import networkx as nx 
import numpy as np
import random
import math


class dataset:
    """ generate random walks and neighbors for  graphsage / unsupervised gcn
    Args:
        path_length: length of random walks
        window_size: window size to sample neighborhood
        num_skips: number of samples to draw from a single walk
        num_neighbor: number of neighbors to sample, for graphsage / unsupervised gcn
    Returns:
        random walk sequences
        1 and 2 order neighbors around a center node
    """
    def __init__(self, params):
        self.path_length = params["TopicGCN"]["path_length"]
        self.num_paths = params["TopicGCN"]["number_paths"]
        self.window_size = params["TopicGCN"]["window_size"]
        self.batch_size = params["TopicGCN"]["batch_size"]
        self.neg_size = params["TopicGCN"]["neg_size"]
        self.num_skips = params["TopicGCN"]["num_skips"]
        self.flag_input_node_feature = params["input_node_feature"]
        self.p = params["TopicGCN"]["p"]
        self.q = params["TopicGCN"]["q"]
        self.data_idx = 0

        dataset_name = params["dataset"] 
 
        if self.flag_input_node_feature == "True":
            # original node features
            self.node_features = np.load(os.path.join("../data/input", dataset_name, "features.npy"))
            # load topic features
            self.node_topic_features = np.load(os.path.join("../data/output", dataset_name, params["TopicModel"]["path_topic_features"]))
        else:
            # if do not input original feature, self.node_features will be topic features
            self.node_features = np.load(os.path.join("../data/output", dataset_name, params["TopicModel"]["path_topic_features"]))
            
        
        self.g, self.num_nodes = self.load_graph(dataset_name)
        self.node2label = self.load_label(dataset_name)
       
        degree_seq_dict = dict(self.g.degree())
        self.degree_seq = [degree_seq_dict[i] for i in range(self.num_nodes)]
        self.neg_sampling_seq = []
        self.preprocess_transition_prob()
        self.random_walks = []

        nodes = list(range(self.num_nodes))
        for _ in range(self.num_paths):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.generate_random_walk(node, self.path_length)
                self.random_walks.append(walk)
            
        self.node_walks = [[] for i in range(self.num_nodes)]
        for w in self.random_walks:
            self.node_walks[w[0]].append(w)
        
        self.types_and_nodes = [[] for i in range(self.num_nodes)]
        self.node_walks = np.array(self.node_walks).astype(int)
      

        self.feature_dim = len(self.node_features[0])
        for i in range(self.num_nodes):
            distr = math.pow(self.degree_seq[i], 0.75)
            distr = math.ceil(distr)
            for _ in range(distr):
                self.neg_sampling_seq.append(i)

        # graphsage neighbor sampling
        self.adj_info = np.zeros((int(self.num_nodes), int(max(self.degree_seq))))
        self.max_degree = max(self.degree_seq)
        for node in range(self.num_nodes):
            neighbors = self.get_neighbor(node)
            if len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, int(self.max_degree), replace = True)
            self.adj_info[node] = neighbors
        self.adj_info = self.adj_info.astype(int)


    def load_graph(self, dataset_name):
        edge_list = []
        with open(os.path.join("../data/input", dataset_name, "edges.txt"), 'r') as infile:
            for line in infile:
                elems = line.rstrip().split(' ')
                src, dst = int(elems[0]), int(elems[1])
                if src == dst:
                    continue
                edge_list.append((src, dst))
        num_nodes = 1 + max(max([u[0] for u in edge_list]), max([u[1] for u in edge_list]))

        g = nx.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        for i, j in edge_list:
            g.add_edge(i, j)
        return g, num_nodes

    def load_label(self, dataset_name):
        node2label = np.zeros((self.num_nodes))
        with open(os.path.join("../data/input/", dataset_name, "labels.txt"), 'r') as infile:
            for line in infile:
                elems = line.rstrip().split(" ")
                node, label = int(elems[0]), int(elems[1])
                node2label[node] = label
        node2label = node2label.astype(int)
        return node2label

    def get_neighbor(self, node):
        """ return neighbor node set of a certain center node
        """
        neighbor = [n for n in self.g.neighbors(node)]
        return neighbor

    def get_alias_edge(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in sorted(self.g.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1 / self.p)
            elif self.g.has_edge(dst_nbr, src):
                # one hop neighbor
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1 / self.q)
        normalize_const = np.sum(unnormalized_probs)
        normalized_probs = [prob/normalize_const for prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)

    
    def preprocess_transition_prob(self):
        alias_nodes = {}
        for nodes in self.g.nodes():
            normalized_probs = [1/self.degree_seq[nodes] for i in range(self.degree_seq[nodes])]
            alias_nodes[nodes] = self.alias_setup(normalized_probs)
            
        alias_edges = {}
        
        for edge in self.g.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return
                
    
    def generate_random_walk(self, begin_node, path_length):
        path = [begin_node]
        while(len(path) < path_length):
            candidates = list(dict(self.g[path[-1]]).keys())
            nextnode = random.choice(candidates)
            path.append(nextnode)
        return path

    def generate_batch(self, batch_size, window_size):
        keys, labels = [], []
        i = 0
        while i < batch_size:
            thiskey = self.random_walks[self.data_idx][0]
            thislabel = self.random_walks[self.data_idx][random.randint(1, self.window_size-1)]
                    
            keys.append(thiskey)  # center node
            labels.append(thislabel) # positive neighbour
            self.data_idx += 1
            self.data_idx %= len(self.random_walks)
            i +=1
        
        negs = self.negative_sampling(keys, labels, self.neg_size)  # negative neighbour

        return np.array(keys).astype(int), np.array(labels).astype(int), np.array(negs).astype(int)


    def negative_sampling(self, keys, labels, neg_size):
        negs = np.zeros((neg_size))

        for j in range(neg_size):
            neg_ = random.choice(self.neg_sampling_seq)
            while (neg_ in labels or neg_ in keys):
                neg_ = random.choice(self.neg_sampling_seq)
            negs[j] = neg_
        return negs    


    def alias_setup(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K).astype(int)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        while len(smaller) >0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q        

 