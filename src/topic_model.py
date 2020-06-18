#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os
import sys
import scipy.io
import scipy.sparse
import math
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix


class GraphAnchorLDA(object):
    def __init__(self, params):
        self.K = params["TopicModel"]["number_topic"]
        self.number_topic = params["TopicModel"]["number_topic"]
        self.path_doc_word = os.path.join("../data/output", params["dataset"], params["Walks"]["path_doc_word"])
        self.path_vocab = os.path.join("../data/output", params["dataset"], params["Walks"]["path_vocab"])
        self.path_word_topic = os.path.join("../data/output", params["dataset"], params["TopicModel"]["path_word_topic"])
        self.path_doc_topic = os.path.join("../data/output", params["dataset"], params["TopicModel"]["path_doc_topic"])
        self.anchor_thresh = params["TopicModel"]["anchor_thresh"]
        self.eps = params["TopicModel"]["eps"]
        self.max_anchors_num = params["TopicModel"]["max_anchors_num"]
        self.flag_max_dim = params["TopicModel"]["flag_max_dim"]
        self.max_features_dim = params["TopicModel"]["max_features_dim"]
        self.path_topic_features = os.path.join("../data/output", params["dataset"], params["TopicModel"]["path_topic_features"])

    def learn_topic_distribution(self):
        self.M = scipy.io.loadmat(self.path_doc_word + ".trunc")['M']  # M [3155*2708] => [word * document]
        # print(M.todense())

        candidate_anchors = []
        # only accept anchors that appear in a significant number of docs
        for i in range(self.M.shape[0]):
            if len(np.nonzero(self.M[i, :])[1]) > self.anchor_thresh:
                candidate_anchors.append(i)
        print ("candidates number:", len(candidate_anchors))

        #forms Q matrix from document-word matrix
        Q = self.generate_Q_matrix(self.M)
        # print ("Q sum is", Q.sum()) #check that Q sum is 1 or close to it

        # find anchors in different topics
        self.anchors = self.select_anchors(Q, candidate_anchors)
        
        # calculate walk-topic distribution 
        self.walk_topic_distribution , topic_likelihoods = self.calculate_walk_topic_distribution(Q, self.anchors)

        np.savetxt(self.path_word_topic, self.walk_topic_distribution)
  

    def generate_Q_matrix(self, M):
        # Given a sparse CSC document matrix M (with floating point entries)
        # comptues the word-word correlation matrix Q
        vocab_size = M.shape[0] # number of words
        num_docs = M.shape[1] # number of documents
        
        diag_M = np.zeros(vocab_size)

        for j in range(M.indptr.size - 1): # M的行数
            # start and end indices for column j
            start = M.indptr[j]
            end = M.indptr[j + 1]
            # end - start = 当前行的元素个数
            
            wpd = np.sum(M.data[start : end]) # 当前行的元素求和
            row_indices = M.indices[start:end]
            # diag_M[row_indices] += M.data[start:end]/(wpd*(wpd-1)) # 如果有某个单词，从来没在某个文档出现过，就会出现nan
            # M.data[start:end] = M.data[start:end]/math.sqrt(wpd*(wpd-1))
            if wpd == 0 or wpd == 1:
                diag_M[row_indices] += 0
                M.data[start : end] = 0
            else:
                diag_M[row_indices] += M.data[start : end] / (wpd * (wpd - 1))
                M.data[start : end] = M.data[start : end] / math.sqrt(wpd * (wpd - 1))
        
        Q = M * M.transpose()/ num_docs # word - word 在每个文档内平均共现的次数
        Q = Q.todense()
        Q = np.array(Q, copy=False)
        diag_M = diag_M / num_docs
        Q = Q - np.diag(diag_M)
        # print ('Sum of entries in Q is ', np.sum(Q))
        return Q


    def select_anchors(self, Q, candidates):
        # select anchors based on walk-walk co-occurrence matrix, through non-negative matrix factorization (NMF)
        Q[Q < 0] = 0.001 # my add
        model = NMF(n_components=self.number_topic, init='random', random_state=0)
        model.fit(Q)
        W = model.components_
        anchors_indices = []
        for topic_anchor in W:
            anchors_indices.append(np.argmax(topic_anchor))

        print("selected indices of anchors:", anchors_indices)
        return anchors_indices


    def calculate_walk_topic_distribution(self, Q, anchors):
        V = Q.shape[0]
        K = len(anchors)
        A = np.matrix(np.zeros((V, K))) # word - topic

        P_w = np.matrix(np.diag(np.dot(Q, np.ones(V))))
        for v in range(V):
            if np.isnan(P_w[v, v]):
                P_w[v, v] = 10**(-16)
        
        # normalize the rows of Q_prime
        for v in range(V):
            Q[v, :] = Q[v, :] / Q[v, :].sum()

        A = np.matrix(np.zeros((V, K)))
        
        X = Q[anchors, :]
        
        for w in range(V):
            y = Q[w, :]
            alpha = self.fast_recover(y, X, w, anchors)
            A[w, :] = alpha

        #rescale A matrix
        #Bayes rule: P(w|z) proportional to P(z|w)P(w)
        A = P_w * A

        #normalize columns of A. This is the normalization constant P(z)
        # word occur_prob in each topic
        topic_likelihoods = A.sum(0)
        for k in range(K): 
            A[:, k] = A[:, k] / A[:,k].sum()
        
        A = np.array(A) 
        print ("walk-topic distribution calculation finished")
        return A, topic_likelihoods


    def fast_recover(self, y, x, v, anchors):
        K = len(anchors)
        alpha = np.zeros(K)
        gap = None
        if v in anchors:
            alpha[anchors.index(v)] = 1
            it = -1
            dist = 0
            stepsize = 0
        else:
            try:
                alpha, it, dist, stepsize = self.KLSolveExpGrad(y, x)
                if np.isnan(alpha).any():
                    alpha = np.ones(K) / K

            except Exception as inst:
                alpha = np.ones(K) / K
                it = -1
                dist = -1
                stepsize = -1
                
        return alpha

    def KLSolveExpGrad(self, y, x):
        c1 = 10**(-4)
        c2 = 0.9
        it = 1 
        
        y = clip(y, 0, 1)
        x = clip(x, 0, 1)

        (K,N) = x.shape
        mask = list(nonzero(y)[0])

        y = y[mask]
        x = x[:, mask]

        x += 10**(-9)
        x /= x.sum(axis=1)[:,newaxis]

        alpha = ones(K)/K

        old_alpha = copy(alpha)
        log_alpha = log(alpha)
        old_log_alpha = copy(log_alpha)
        proj = dot(alpha,x)
        old_proj = copy(proj)

        log_y = log(y)
        new_obj = KL(y,log_y, proj)
        y_over_proj = y / proj
        grad = -dot(x, y_over_proj.transpose())
        old_grad = copy(grad)

        stepsize = 1
        decreasing = False
        repeat = False
        gap = float('inf')

        while 1:
            eta = stepsize
            old_obj = new_obj
            old_alpha = copy(alpha)
            old_log_alpha = copy(log_alpha)

            old_proj = copy(proj)
            it += 1
            log_alpha -= eta * grad #take a step

            log_alpha -= logsum_exp(log_alpha)  #normalize

            #compute new objective
            alpha = exp(log_alpha)
            proj = dot(alpha,x)
            new_obj = KL(y,log_y,proj)
            if new_obj < self.eps:
                break

            grad_dot_deltaAlpha = dot(grad, alpha - old_alpha)
            assert (grad_dot_deltaAlpha <= 10**(-9))
            if not new_obj <= old_obj + c1*stepsize*grad_dot_deltaAlpha: #sufficient decrease
                stepsize /= 2.0 #reduce stepsize
                if stepsize < 10**(-6):
                    break
                alpha = old_alpha 
                log_alpha = old_log_alpha
                proj = old_proj
                new_obj = old_obj
                repeat = True
                decreasing = True
                continue

            
            #compute the new gradient
            old_grad = copy(grad)
            y_over_proj = y/proj
            grad = -dot(x, y_over_proj)

            if not dot(grad, alpha - old_alpha) >= c2 * grad_dot_deltaAlpha and not decreasing: #curvature
                stepsize *= 2.0 #increase stepsize
                alpha = old_alpha
                log_alpha = old_log_alpha
                grad = old_grad
                proj = old_proj
                new_obj = old_obj
                repeat = True
                continue

            decreasing= False
            lam = copy(grad)
            lam -= lam.min()
            
            gap = dot(alpha, lam)
            convergence = gap
            if (convergence < self.eps):
                break

        return alpha, it, new_obj, stepsize



    def select_topic_features(self):
        key_structures = self.find_key_structures(self.walk_topic_distribution)
        # print(len(key_structures))

        M = self.M.toarray().T # document * word

        # generate lda features based on key_structures and walk_topic_distribution
        feats = np.zeros((M.shape[0], len(key_structures)))
        for i in range(M.shape[0]):
            for (idx, val) in enumerate(key_structures):
                if M[i][val] > 0:
                    feats[i][idx] = 1.0

        # 如果出现某一行全是0的情况，就随机给一个特别小的值
        for i in range(len(feats)):
            if np.all(feats[i] == 0.0): # all
                feats[i][0] = 0.00001


        # if the max dim of lda features is restricted
        if self.flag_max_dim: 
            np.save(self.path_topic_features, feats[ : , : self.max_features_dim])
            return feats[ : , : self.max_features_dim]
        
        np.save(self.path_topic_features, feats)
        return feats


    def find_key_structures(self, walk_topic_distribution):
        X = walk_topic_distribution
        X_var = np.var(X, axis=1)

        key_structures = []

        for i in range(len(X_var)):
            if np.any(X[i] == 0.0): # 方差很大, 有一列为0
                key_structures.append(i)


        if self.flag_max_dim and len(key_structures) > self.max_features_dim:
            return key_structures

        if len(key_structures) < self.max_anchors_num:
            return key_structures

        argsorted = np.argsort(X_var) # 大的在后面
        for i in range(number - top_num - 1, number):
            if argsorted[i] not in key_structures:
                key_structures.append(argsorted[i])

        return key_structures


    def calculate_node_topic_distribution(self):
        M = self.M.toarray().T 
        walk_topic_distribution_reverse = np.linalg.pinv(self.walk_topic_distribution.T)
        node_topic_distribution = np.matmul(M, walk_topic_distribution_reverse) 
        np.savetxt(self.path_topic_topic, node_topic_distribution)
        print ("node-topic distribution calculation finished")
