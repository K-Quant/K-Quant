import itertools
import pandas as pd
from collections import defaultdict
from itertools import islice, chain
import re
from datetime import datetime
from datetime import timedelta
import networkx as nx
import numpy as np
import os
import dateutil.parser
from scipy.sparse import csr_matrix
import json 

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d

def build_links(kg_df):    
    links = []
    ts = []
    for i in range(len(kg_df)):
        timesting = kg_df['time'][i]
        timestamp = getDateTimeFromISO8601String(timesting)
        ts.append(timestamp)
        links.append((kg_df['e1'][i],kg_df['e2'][i],timestamp))
    return links,ts

def subgraph(links,ts,SLICE_MONTHS,START_DATE,MAX_DATE):
    slices_links = defaultdict(lambda : nx.MultiGraph())
    slices_features = defaultdict(lambda : {})

    print ("Start date", START_DATE)
    slice_id = -1
    time_slides ={}
    # Split the set of links in order by slices to create the graphs.
    for (a, b, time) in links:
        prev_slice_id = slice_id
        
        datetime_object = time
        if datetime_object > MAX_DATE:
            months_diff = (MAX_DATE - START_DATE).days//30  # 
            
        else:
            months_diff = (datetime_object - START_DATE).days//30

        slice_id = months_diff // SLICE_MONTHS      
        slice_id = max(slice_id, 0)

        if str(slice_id) not in time_slides:
            time_slides[str(slice_id)] = []
        time_slides[str(slice_id)].append(time)

        if slice_id == 1+prev_slice_id and slice_id ==0:
            slices_links[slice_id] = nx.MultiGraph()
            
        if slice_id == 1+prev_slice_id and slice_id > 0:
            slices_links[slice_id] = nx.MultiGraph()
            slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
            slices_links[slice_id].add_edges_from(slices_links[slice_id-1].edges(data=True)) 
            ## ***** keep the edges from the last graph
            
        if a not in slices_links[slice_id]:
            slices_links[slice_id].add_node(a)
        if b not in slices_links[slice_id]:
            slices_links[slice_id].add_node(b)
        slices_links[slice_id].add_edge(a,b, date=datetime_object)
        


    for slice_id in slices_links:
        print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
        print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
        temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
        slices_features[slice_id] = {}
        for idx, node in enumerate(slices_links[slice_id].nodes()):
            slices_features[slice_id][node] = temp[idx]

    return slices_features,slices_links,time_slides





def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    all_nodes = list(set(all_nodes))
    #print(slices_graph[1].nodes())
    print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    # print ("Total # nodes", len(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1

    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x],code = x)
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes(data=True)))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            # print(slices_features[slice_id][idx_node[x]])
            # features_remap.append(slices_features[slice_id][idx_node[x]])
            features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    
    return slices_graph_remap, slices_features_remap, idx_node,node_idx

dykg = pd.read_csv('data/FinKG/dytuple.csv')

static_kg = pd.read_csv('data/FinKG/SW_csi_22.csv')
static_kg['time'] = min(dykg['time'])

column_order = [ 'e1', 'e2','time']
whole_kg =  pd.concat([dykg[column_order], static_kg[column_order]], ignore_index=True)
# whole_kg['time'] = pd.to_datetime(whole_kg['time'])
whole_kg = whole_kg.sort_values(['time'])
whole_kg = whole_kg.reset_index(drop=True)
links,ts = build_links(whole_kg)
SLICE_WEEKS,START_DATE,MAX_DATE = 1, min(ts) + timedelta(days=0), max(ts)-timedelta(days=0)
slices_features,slices_links,time_slides = subgraph(links,ts,SLICE_WEEKS,START_DATE,MAX_DATE)
slices_links_remap, slices_features_remap, idx_node,node_idx = remap(slices_links, slices_features)

slice_table = []
for i in time_slides.keys():
    slice_table.append([min(time_slides[i]),max(time_slides[i])])
time_index_df = pd.DataFrame(slice_table,columns=['start','end'])


path = "data/FinKG"
time_index_df.to_csv('slice_table_dyge.csv')

np.savez(path + '/graphs_csi22_1_no.npz', graph=slices_links_remap)
np.savez(path + '/features_csi22_1_no.npz', feats=slices_features_remap)
print('Successfully save the graphs and feature matrices in ' + path)
