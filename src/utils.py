
import os
import re
import json
import numpy as np
import networkx as nx


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

    return params



'''
def cos_sim(self, vec_a, vec_b):
    res1, a_norm, b_norm = 0.0, 0.0, 0.0
    length = len(vec_a)
    for i in range(length):
        res1 += vec_a[i] * vec_a[i] 
        a_norm += vec_a[i] * vec_a[i]
        b_norm += vec_b[i] * vec_b[i]

    a_norm = math.sqrt(a_norm)
    b_norm = math.sqrt(b_norm)

    return res1 / (a_norm * b_norm)
'''