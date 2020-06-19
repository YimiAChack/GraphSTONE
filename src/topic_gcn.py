#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import time
import threading
import queue
import tensorflow as tf 
import numpy as np
import networkx as nx 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from dataset import *


class StructuralTopicGCN:
    """ Structural Topic GCN
    Args:
        neg_size: negative sampling size
        num_steps: steps to train
        embedding_dims: the size of node embeddings
        num_neighbor: number of neighbors to sample, for graphsage / unsupervised gcn
    Returns:
        node embeddings
    """
    def __init__(self, params):       
        self.dataset_name = params["dataset"]        
        self.window_size = params["TopicGCN"]["window_size"]
        self.neg_size = params["TopicGCN"]["neg_size"]
        self.batch_size = params["TopicGCN"]["batch_size"]
        self.learning_rate = params["TopicGCN"]["learning_rate"]
        self.num_steps = params["TopicGCN"]["max_training_steps"]
        self.hidden_dim = params["TopicGCN"]["hidden_dim"]
        self.num_neighbor = params["TopicGCN"]["num_neighbor"]
        self.p = params["TopicGCN"]["p"]
        self.q = params["TopicGCN"]["q"]
        self.flag_input_node_feature = params["input_node_feature"]

        if params["input_node_feature"]: # !!!
            self.embedding_dims = params["TopicGCN"]["embedding_dims"] // 2 # combined model, finally will concate
        else:
            self.embedding_dims = params["TopicGCN"]["embedding_dims"]

        self.save_path = os.path.join("../data/output", self.dataset_name, "model") 
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.sess = tf.Session()
        self.start_time = time.time()
        
        self.Dataset = dataset(params)
        self.num_nodes = self.Dataset.num_nodes
        self.feature_dims = len(self.Dataset.node_features[0])
        self.node_features = self.Dataset.node_features
        self.node_topic_features = self.Dataset.node_topic_features
        self.feature_dims_topic = len(self.Dataset.node_topic_features[0])

        self.build_model()
        print("[%.2fs] Finish sampling, begin training ..." % (time.time() - self.start_time))
        # self.saver = tf.train.Saver(max_to_keep = 5)
    

    def build_model(self):
        """
            walk: key, label, neg
            node: key, label, neg
            walks are dealt with directly while nodes will need sampling
        """
        self.batch_keys = tf.placeholder(tf.int32, [None])
        self.batch_labels = tf.placeholder(tf.int32, [None])
        self.batch_negs = tf.placeholder(tf.int32, [None])
        self.batch_input = tf.placeholder(tf.int32, [None])
        self.input_size = tf.placeholder(tf.int32)

        # sample一样，只是特征不一样
        samples_keys = self.sample(self.batch_keys, self.num_neighbor, self.batch_size)
        samples_labels = self.sample(self.batch_labels, self.num_neighbor, self.batch_size)
        samples_negs = self.sample(self.batch_negs, self.num_neighbor, self.neg_size)
        samples_output = self.sample(self.batch_input, self.num_neighbor, self.input_size)


        if self.flag_input_node_feature:
            output_key = self.aggregate(samples_keys, self.batch_size, self.feature_dims, self.node_features, False)
            output_label = self.aggregate(samples_labels, self.batch_size, self.feature_dims, self.node_features, False)
            output_neg = self.aggregate(samples_negs, self.neg_size, self.feature_dims, self.node_features, False)
            output_normal = self.aggregate(samples_output, self.input_size, self.feature_dims, self.node_features, False)
            
            output_key_topic = self.aggregate(samples_keys, self.batch_size, self.feature_dims_topic, self.node_topic_features, True)
            output_label_topic = self.aggregate(samples_labels, self.batch_size, self.feature_dims_topic, self.node_topic_features, True)
            output_neg_topic = self.aggregate(samples_negs, self.neg_size, self.feature_dims_topic, self.node_topic_features, True)
            output_topic = self.aggregate(samples_output, self.input_size, self.feature_dims_topic, self.node_topic_features, True)

            output_key_combine = self.combine(output_key, output_key_topic, self.batch_size)
            output_label_combine = self.combine(output_label, output_label_topic, self.batch_size)
            output_neg_combine = self.combine(output_neg, output_neg_topic, self.neg_size)
            self.output = self.combine(output_normal, output_topic, self.input_size)# topic 和normal vector拼起来
            self.loss = self.compute_loss(output_key_combine, output_label_combine, output_neg_combine)
        else:
            output_key = self.aggregate(samples_keys, self.batch_size, self.feature_dims, self.node_topic_features, False)
            output_label = self.aggregate(samples_labels, self.batch_size, self.feature_dims, self.node_topic_features, False)
            output_neg = self.aggregate(samples_negs, self.neg_size, self.feature_dims, self.node_topic_features, False)
            output = self.aggregate(samples_output, self.input_size, self.feature_dims, self.node_topic_features, False)
            self.output = output
            self.loss = self.compute_loss()

        self.optim = tf.train.AdamOptimizer(self.learning_rate)
      
        # Clipping
        # grads_and_vars = self.optim.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #     for grad, var in grads_and_vars]
        # self.opt_op = self.optim.apply_gradients(clipped_grads_and_vars)

        # No clipping
        self.opt_op = self.optim.minimize(self.loss)


    def combine(self, output1, output2, len):
        with tf.variable_scope("combine", reuse = tf.AUTO_REUSE): #???
            combine_weight = tf.get_variable("combined_W", [self.embedding_dims * 2, self.embedding_dims * 2], dtype = tf.float64, 
                        initializer = tf.contrib.layers.xavier_initializer())
            combine_bias = tf.get_variable("combined_bias", [self.embedding_dims * 2], dtype = tf.float64, 
                        initializer = tf.constant_initializer(0.01))
       
        output_merge = tf.tanh(tf.concat([output1, output2], 1))
        output_merge = tf.matmul(output_merge, combine_weight) + combine_bias
        output_merge = tf.nn.l2_normalize(output_merge, 1)
        output1 = tf.nn.l2_normalize(output1, 1)
        output_merge = tf.concat([output1, output_merge], 1)
        return output_merge 


    def compute_loss(self, output_key, output_label, output_neg):
        pos_aff = tf.reduce_sum(tf.multiply(output_key, output_label), axis = 1)
        neg_aff = tf.einsum("ij,kj->ik", output_key, output_neg)
        likelihood = tf.log(tf.sigmoid(pos_aff) + 1e-6) + tf.reduce_sum(tf.log(1 - tf.sigmoid(neg_aff) + 1e-6), axis = 1)        
        return -tf.reduce_mean(likelihood)

      
    def sampleNeighbor(self, batch_nodes, num_samples): 
        adj_lists = tf.nn.embedding_lookup(self.Dataset.adj_info, batch_nodes) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        neigh_nodes = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        
        return  tf.squeeze(neigh_nodes) # 删除所有为1的维度


    def sample(self, inputs, num_sample, input_size):
        samples = [inputs]
        support_size = input_size

        for k in range(2): # 2-order neighbours
            support_size *= num_sample
            nodes = self.sampleNeighbor(samples[k], num_sample)
            samples.append(tf.reshape(nodes, [support_size]))

        return samples


    def aggregate(self, sample_nodes, input_size, feature_dims, features, flag_topic):
        dims = [feature_dims, self.hidden_dim, self.embedding_dims]
        support_size = [1, self.num_neighbor, self.num_neighbor**2]
        hidden_nodes = [tf.nn.embedding_lookup(features, nodes) for nodes in sample_nodes]

        # two layer
        for layer in range(2):
            input_dim, output_dim = dims[layer], dims[layer + 1]
            scope_name = "aggregator_topic_" if flag_topic else "aggregator_"
            with tf.variable_scope(scope_name + str(layer), reuse = tf.AUTO_REUSE):
                weight_self = tf.get_variable("weight_self", [input_dim, output_dim], dtype = tf.float64, 
                    initializer = tf.contrib.layers.xavier_initializer())
                weight_neigh = tf.get_variable("weight_neigh", [input_dim, output_dim], dtype = tf.float64, 
                    initializer = tf.contrib.layers.xavier_initializer())
                next_hidden = [] 
                for hop in range(2 - layer):
                    neigh_dims = [input_size * support_size[hop], self.num_neighbor, dims[layer]]
                    neigh_vecs = tf.reshape(hidden_nodes[hop + 1], neigh_dims)
                    neigh_mean = tf.reduce_mean(neigh_vecs, axis = 1)
                    from_neighs = tf.matmul(neigh_mean, weight_neigh)
                    from_self = tf.matmul(hidden_nodes[hop], weight_self)
                    if layer != 1:
                        final = tf.nn.relu(from_neighs + from_self)
                    else: 
                        final = from_neighs + from_self
                    next_hidden.append(final)
                hidden_nodes = next_hidden
        return hidden_nodes[0]


    def train(self):
        self.sess.run(tf.global_variables_initializer())

        def load_batch(q):
            while 1:
                batchkeys, batchlabels, batchnegs = self.Dataset.generate_batch(self.batch_size, self.window_size)
                q.put((batchkeys, batchlabels, batchnegs))
        
        q = queue.Queue(maxsize = 5)
        t = threading.Thread(target = load_batch, args = [q])
        t.daemon = True
        t.start()
        
        losses = []
        for i in range(self.num_steps):
            keys, labels, negs = q.get()
            _, batch_loss = self.sess.run([
                self.opt_op, 
                self.loss, 
            ], 
            feed_dict = {
                self.batch_keys: keys, 
                self.batch_labels: labels, 
                self.batch_negs: negs
            })
            losses.append(batch_loss)
            if i and i % 100 == 0:
                print("[%.2fs] After %d iters, loss  on training is %.4f."%(time.time() - self.start_time, i, np.mean(losses)))
                losses = []
            if i and i % 500 == 0:
                self.evaluation()
                self.save_embeddings(i)
                # self.saver.save(self.sess, "model/topical_gcn", global_step=i)
    
    
    def get_full_embeddings(self):
        self.embedding_array = np.zeros((self.Dataset.num_nodes, self.embedding_dims * 3))
        batch_size = 100
        for i in range(self.Dataset.num_nodes // batch_size + 1):
            if i != self.Dataset.num_nodes // batch_size:
                batchnode = np.arange(100*i, 100*i+100)
                batch_embeddings = self.sess.run([self.output], feed_dict = {
                    self.batch_input: batchnode, 
                    self.input_size: 100
                })
                self.embedding_array[100*i : 100*i + 100] = batch_embeddings[0]
            else:
                batchnode = np.arange(100*i, self.num_nodes)
                batch_embeddings = self.sess.run([self.output], feed_dict = {
                    self.batch_input: batchnode, 
                    self.input_size: self.num_nodes - 100*i
                })
                self.embedding_array[100*i : self.num_nodes] = batch_embeddings[0]
        return self.embedding_array

    
    def save_embeddings(self, step):
        np.save(os.path.join(self.save_path, str(step)), arr = self.embedding_array)
        print("Embedding saved for step #%d" %step)

    def evaluation(self):
        # the evaluate_model is only used for tracking the training process and cannot be used 
        # for formal model evaluation
        self.get_full_embeddings()
        macros = []
        micros = []
        for _ in range(5):
            validation_indice = random.sample(range(self.num_nodes), self.num_nodes // 5)
            train_indice = [i for i in range(self.num_nodes) if i not in validation_indice]
            train_feature = self.embedding_array[train_indice]
            train_label = self.Dataset.node2label[train_indice]
            validation_feature = self.embedding_array[validation_indice]
            validation_label = self.Dataset.node2label[validation_indice]

            clf = LogisticRegression(solver='lbfgs', max_iter=800)
            clf.fit(train_feature, train_label)
            predict_label = clf.predict(validation_feature)
            macro_f1 = metrics.f1_score(validation_label, predict_label, average= "macro")
            micro_f1 = metrics.f1_score(validation_label, predict_label, average = "micro")
            macros.append(macro_f1)
            micros.append(micro_f1)
        print("Node classification macro f1: %.4f"%np.mean(macros))
        print("Node classification micro f1: %.4f"%np.mean(micros))   
