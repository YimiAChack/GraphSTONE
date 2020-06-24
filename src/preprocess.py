#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import scipy.io
import numpy as np
import networkx as nx
import utils


class PreProcess(object): 
    """ generate topic concepts, for the input of GraphAnchorLDA
    Args:
        number_paths: number of random walks started from a center node
        path_length: length of random walks
        anonymous_walk_max_length: max length of anonymous walks
    Returns:
        Co-occurrence of graph "word" (anonymous walk) and "document" (neighborhoods around a center node)
    """
    def __init__(self, params): 
        self.G = nx.read_edgelist(os.path.join("../data/input", params["dataset"], "edges.txt"), create_using = nx.Graph())
        self.path_out_docword = os.path.join("../data/output", params["dataset"], params["PreProcess"]["path_doc_word"])
        self.path_out_vocab = os.path.join("../data/output", params["dataset"], params["PreProcess"]["path_vocab"])
        self.number_paths = params["PreProcess"]["number_paths"]
        self.anonymous_walk_max_length = params["PreProcess"]["anonymous_walk_max_length"]
        self.path_length = params["PreProcess"]["path_length"]
        self.return_prob = params["PreProcess"]["return_prob"]
        print("start generating topic concepts")

        
    def generate_topic_concepts(self):
        """generate topic concepts, for the input of GraphAnchorLDA
        """

        # generate random walks and corresponding anonymous walks
        random_walks = self.generate_random_walks()
        anonymous_walks = self.generate_anonymous_walks(random_walks)

        # build word-document occurrence
        self.generate_word_document(anonymous_walks, self.path_out_docword, self.path_out_vocab)

        # filer low frequency local-structures, and trans the format
        sparse_matrix = self.trans_matrix_sparse(self.path_out_docword)
        self.truncate_vocabulary(sparse_matrix, self.path_out_vocab, self.path_out_docword + ".trunc", self.path_out_vocab + ".trunc" , cutoff = 1)


    def generate_random_walks(self, rand=random.Random(0)):
        """generate random walks
        """
        walks = list()
        for node in list(self.G.nodes()):
            for _ in range(self.number_paths):
                walks.append(self.random_walk(rand=rand, start=node))
        return walks


    def generate_anonymous_walks(self, random_walks):
        """generate anonymous walks, based on random walks
        """
        anonymous_walks = [[] for i in range(self.G.number_of_nodes())]  # Take each node as the center node

        for random_walk_seq in random_walks:
            anonymous_walk_seq = self.random_to_anonymous_walk(random_walk_seq)
            if 2 < len(anonymous_walk_seq) <= self.anonymous_walk_max_length: # select seq with specified length
                center_node = int(random_walk_seq[0])
                anonymous_walks[center_node].append(anonymous_walk_seq) 
                # if not filter_any(anonym_walk):
                #     anonymous_walks[int(w[0])].append(anonym_walk)
        return anonymous_walks


    def random_walk(self, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start is None:
            print("random walk need a start node!")
        path = [start]

        cur_path_length = random.randint(4, self.path_length + 1)
        while len(path) < cur_path_length:
            cur = path[-1]
            if len(self.G.neighbors(cur)) > 0:
                if rand.random() >= self.return_prob:
                    # print(G.neighbors(cur)) # !!!
                    path.append(rand.choice(self.G.neighbors(cur)))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


    def random_to_anonymous_walk(self, random_walk_seq):
        """convert a random walk sequence to an anonymous walk
        """
        cnt = 0
        node_cnt = dict()
        anonymous_walk_seq = []
        for node in random_walk_seq:
            if node not in node_cnt:
                cnt += 1
                node_cnt[node] = cnt
            anonymous_walk_seq.append(node_cnt[node])

        return anonymous_walk_seq


    def trans_matrix_sparse(self, input_matrix):
        """transform input matrix to sparse matrix
        """
        infile = open(input_matrix, "r")
        num_docs = int(infile.readline())
        num_words = int(infile.readline())

        sparse_matrix = scipy.sparse.lil_matrix((num_words, num_docs))

        for l in infile:
            d, w, v = [int(x) for x in l.split()]
            sparse_matrix[w, d] = v

        return sparse_matrix


    def truncate_vocabulary(self, input_sparse_matrix, input_full_vocab, output_matrix, output_vocab, cutoff):
        """Read in the vocabulary, and build a symbol table mapping words to indices
        """
        table = dict()
        numwords = 0
        for line in open(input_full_vocab, 'r'):
            table[line.rstrip()] = numwords
            numwords += 1


        remove_word = [False] * numwords

        M = input_sparse_matrix.tocsr()
        new_indptr = np.zeros(M.indptr.shape[0], dtype=np.int32)
        new_indices = np.zeros(M.indices.shape[0], dtype=np.int32)
        new_data = np.zeros(M.data.shape[0], dtype=np.float64)

        indptr_counter, data_counter = 1, 0
        for i in range(M.indptr.size - 1):
            # if this is not a stopword
            if not remove_word[i]:
                # start and end indices for row i
                start = M.indptr[i]
                end = M.indptr[i + 1]
                        
                # if number of distinct documents that this word appears in is >= cutoff
                if (end - start) >= cutoff:
                    new_indptr[indptr_counter] = new_indptr[indptr_counter - 1] + end - start
                    new_data[new_indptr[indptr_counter -  1] : new_indptr[indptr_counter]] = M.data[start : end]
                    new_indices[new_indptr[indptr_counter - 1] : new_indptr[indptr_counter]] = M.indices[start : end]
                    indptr_counter += 1
                else:
                    remove_word[i] = True 

        new_indptr = new_indptr[0 : indptr_counter]
        new_indices = new_indices[0 : new_indptr[indptr_counter - 1]]
        new_data = new_data[0 : new_indptr[indptr_counter - 1]]


        M = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr))
        M = M.tocsc()
        scipy.io.savemat(output_matrix, {'M' : M}, oned_as='column')

        print ('New number of words is ', M.shape[0])
        print ('New number of documents is ', M.shape[1])

        # Output the new vocabulary
        output = open(output_vocab, 'w')
        row = 0
        with open(input_full_vocab, 'r') as file:
            for line in file:
                if not remove_word[row]:
                    output.write(line)
                row += 1
        output.close()


    def generate_word_document(self, anonymous_walks, outfile_docword, outfile_vocab):
        """generate word-document, for the input of LDA
        """
        anonymous_dict = dict() 
        idx = 0
        anonymous_walks_idx = [[] for _ in range(self.G.number_of_nodes())] 
        
        out_vocal = open(outfile_vocab, "w") 

        # use an index to represent one anonymous walk seq
        for i in range(self.G.number_of_nodes()):
            for w in anonymous_walks[i]:
                # w = [1,2,3]
                w = [str(_) for _ in w]
                w = " ".join(w) # "1 2 3"

                if w in anonymous_dict:
                    anonymous_walks_idx[i].append(anonymous_dict[w])
                    # one document: ['0', '0', '5'], '0' and '5' present different anonymous walks
                else:
                    anonymous_dict[w] = str(idx)
                    idx += 1
                    out_vocal.write(w + "\n")
                    anonymous_walks_idx[i].append(anonymous_dict[w])


        #count number of word occurrence in each document
        anonymous_walks_cnt = [[] for _ in range(self.G.number_of_nodes())] 
        for i in range(self.G.number_of_nodes()):
            cnt = dict()
            for w in anonymous_walks_idx[i]: # w is the idx of anonymous_walk_seq
                if w not in cnt:
                    cnt[w] = anonymous_walks_idx[i].count(w)
            anonymous_walks_cnt[i] = cnt


        out = open(outfile_docword, "w")
        out.write(str(self.G.number_of_nodes()) + "\n")
        out.write(str(len(anonymous_dict)) + "\n")

        for i in range(self.G.number_of_nodes()):
            cnt = anonymous_walks_cnt[i]
            for word in cnt.keys():
                out.write(str(i) + " " + word + " " + str(cnt[word]) + "\n")
        
        out.close()

