from __future__ import print_function
import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from random_walk import Graph_RandomWalk
from temporal_random_walk import TemporalRandomWalk

flags = tf.app.flags
FLAGS = flags.FLAGS


def to_one_hot(labels, N, multilabel=False):
    """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


"""Random walk-based pair generation."""

def run_temporal_random_walks(graph):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    walk_len = FLAGS.walk_len
    num_walks = FLAGS.num_walks
    G = TemporalRandomWalk(graph)
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("#first sampled pairs: {}".format(pairs_cnt))
    return pairs


def run_temporal_random_walks_1(graph):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    walk_len = 80
    num_walks = 10
    G = TemporalRandomWalk(graph)
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs



# def load_graphs(dataset_str):
#     """Load graph snapshots given the name of dataset"""
#     graphs = np.load("../data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True,fix_imports=True)['graph']
#     print("Loaded {} graphs ".format(len(graphs)))
#     adj_matrices = map(lambda x: nx.adjacency_matrix(x), graphs)
#     return graphs, adj_matrices


# graphs,_ = load_graphs("Enron_new")
# print("Computing training pairs ...")

# pairs = run_temporal_random_walks(graphs[1])

    
