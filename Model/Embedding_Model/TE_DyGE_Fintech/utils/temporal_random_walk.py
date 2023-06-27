
import numpy as np
import random
from typing import Tuple
import networkx as nx
import scipy.stats
from numpy.random import choice

class TemporalRandomWalk():

    def __init__(self, nx_G):
        self.G = nx_G

    def sampling_edge(self):
        
        edges = list(self.G.edges(data=True))
        edges = sorted(edges, key = lambda x:x[2].values()[0])
        timestamps = [e[2].values()[0] for e in edges]
        # linear distribution 
        r = scipy.stats.rankdata(timestamps, 'dense')
        probs = r / np.sum(r)
        # Sample edge over distribution
        sampled_t = random.choice(timestamps)

        valid_e = [e for e in edges if e[2].values()[0]==sampled_t]

        sampled_e = random.choice(valid_e)

        return sampled_e


    def tr_Walk(self, walk_length):

        start_edges = self.sampling_edge()
        t = start_edges[-1].values()[0]
        j = start_edges[0]
        walk = [j]
        # Iterates until length of walk is reached 
        for _ in range(walk_length - 2):
            # Compute in-between periods of time between source node and temporal neighbors
            cur_nbrs = list(self.G.edges(j,data=True))
            cur_nbrs = sorted(cur_nbrs, key = lambda x:x[2].values()[0])
            timestamps = np.array([e[2].values()[0] for e in cur_nbrs])
            valid_times = timestamps[timestamps >= t]
            delta_times = valid_times - t
            sampled_t = random.choice(valid_times)
            neighbs = [v for u,v,t in self.G.edges(data=True) if t.values()[0] == sampled_t]
            # # sampled_n = random.choices(neighbs)[0]    <- using distribution
            sampled_n = random.choice(neighbs)
            walk.append(sampled_n)
            # # Update current nodes and timestamp
            j = sampled_n
            t = sampled_t
        return walk

    # TODO: using distribution to sample the neighbors

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            print('simulate walks',walk_iter)
            a = self.tr_Walk(walk_length)
            walks.append(a)

        return walks

    