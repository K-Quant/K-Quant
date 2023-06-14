import pandas as pd 
from stellargraph import StellarGraph
from stellargraph.data import UniformRandomMetaPathWalk,BiasedRandomWalk
from gensim.models import Word2Vec
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import datasets
from IPython.display import display, HTML




#  StellarDiGraph  
"""
cn_company_node
"""
def build_graph(monthdata,df_csi):
    column_order = [ 'e1', 'e2']
    combined_df = pd.concat([df_csi[column_order], monthdata[column_order]], ignore_index=True)
    combined_df = combined_df.rename(columns={'e1': 'source', 'e2': 'target'})

    combined_column = pd.concat([combined_df['source'], combined_df['target']])
    unique_values = combined_column.drop_duplicates()
    company_node = pd.DataFrame(unique_values,columns=['code'])
    company_node = company_node.set_index("code")

    company_graph = StellarGraph(
    company_node, combined_df)
    print(company_graph.info())
    return company_graph


def walk_setting(walk_length,n_rw,company_graph):
    """maximum length of a random walk to use throughout this notebook"""
    # Create the random walker
    rw = BiasedRandomWalk(company_graph)   ##########
    # rw = UniformRandomMetaPathWalk(company_graph)

    # walks = rw.run(
    #     nodes=list(company_graph.nodes()),  # root nodes
    #     length=walk_length,  # maximum length of a random walk
    #     n=20,  # number of random walks per root node
    #     metapaths=metapaths,  # the metapaths
    # )
    walks = rw.run(
      nodes=list(company_graph.nodes()), # root nodes
      length=walk_length,  # maximum length of a random walk
      n=n_rw       # number of random walks per root node 
)
    print("Number of random walks: {}".format(len(walks)))
    return walks


def run_save(time_steps,walks,company_graph,dim):
    model = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1, workers=2)  ## set dimension
    print('shape:',model.wv.vectors.shape)  # ?-dimensional vector for each node in the graph
    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )
    # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = [company_graph.node_type(node_id) for node_id in node_ids]
    node_embeddings = node_embeddings.tolist()
    embeddings_df = pd.DataFrame({
        'label':node_targets,
        'name':node_ids,
        'embeddings':node_embeddings
    })

    # Save the embeddings as an npz file
    embeddings = np.array(embeddings_df['embeddings'].tolist())
    np.savez('output/newcsi/kgeef_dim'+ str(dim) + '_W' + str(time_steps) + '.npz', embeddings=embeddings)
    print('successfully saved')

    # Save id2stock
    name_dict = {idx: name for idx, name in enumerate(embeddings_df['name'].unique())}
    with open('output/newcsi/kgeef_id2stock_W'+ str(time_steps)+'.pkl', 'wb') as file:
        pickle.dump(name_dict, file)
    

## preprocess
df = pd.read_csv('../data/newcsi/dytuple.csv')
df_csi = pd.read_csv('../data/newcsi/SW_csi_22.csv')

## split graph for a specific month
df['time'] = pd.to_datetime(df['time'])  
df = df.sort_values('time')
print('start date',df['time'].min())
print('end date',df['time'].max())

grouped_data = {}
cumulative_data = pd.DataFrame()
time_index = []
for i, (name, group) in enumerate(df.groupby(pd.Grouper(key='time', freq='W')), start=0):
    cumulative_data = pd.concat([cumulative_data, group])
    grouped_data[i] = cumulative_data
    time_index.append([min(cumulative_data['time']),max(cumulative_data['time'])])

time_index_df = pd.DataFrame(time_index,columns=['start','end'])
time_index_df.to_csv('output/newcsi/time_index.csv')

## walk setting 
walk_length = 20
n_rw = 10
dim = 8

for i in range(len(grouped_data)):
    company_graph = build_graph(grouped_data[i],df_csi)
    walks = walk_setting(walk_length,n_rw,company_graph)
    run_save(i,walks,company_graph,dim)



