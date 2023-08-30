from __future__ import division
from __future__ import print_function

import torch
import dgl

from .base import GraphExplainer


class AttentionX(GraphExplainer):
    def __init__(self, graph_model, num_layers, device):
        super(AttentionX, self).__init__(graph_model, num_layers, device)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        #self.target_ntype = 's'


    def explain(self, full_model, graph, stkid):
        #target_ntype = full_model.target_type
        #target_ntype = self.target_ntype
        dataloader = dgl.dataloading.DataLoader(graph,
                                                     torch.Tensor([stkid]).type(torch.int64).to(self.device),
                                                     self.sampler,
                                                    batch_size=1, shuffle=False, drop_last=False)


        neighbors = None
        for neighbors, _, _ in dataloader:
            break

        target_id = stkid

        g_c = dgl.node_subgraph(graph, neighbors) # induce the computation graph
        attn = full_model.get_attention(g_c)
        new_target_id = (g_c.ndata['_ID'].tolist()).index(target_id)
        g_c = g_c.to(self.device)

        if self.graph_model == 'homograph':
            for l in range(self.num_layers):
                attn[l] = torch.mean(attn[l], dim=1).squeeze(dim=1).tolist()  # summarize multi-head attention

            node_attn = {self.num_layers: {new_target_id: 1}}
            for l in range(self.num_layers - 1, -1, -1):  # assign attention to nodes iteratively from top layer
                node_attn[l] = {}  # src: attn_score
                src, dst = g_c.edges()
                for j, (ss, dd) in enumerate(zip(src.tolist(), dst.tolist())):
                    if (dd in node_attn[l + 1]):  # connected to top layer nodes
                        node_attn[l][ss] = attn[l][j] * node_attn[l + 1][dd]

            # sum up for each node
            ne_attn = {}
            new2old = {}
            for j, n in enumerate(neighbors.tolist()):
                ne_attn[n] = 0
                new2old[j] = n

        else:
            for l in range(self.num_layers):
                for tp in g_c.etypes:
                    if len(attn[l][tp].shape)==3: # summarize multi-head attention
                        attn[l][tp] = torch.mean(attn[l][tp], dim=1).squeeze(dim=1).tolist()
                    else:
                        attn[l][tp] = attn[l][tp].squeeze(dim=1).tolist()
            node_attn = {self.num_layers: {new_target_id: 1}}
            for l in range(self.num_layers - 1, -1, -1):  # assign attention to nodes iteratively from top layer
                node_attn[l] = {}
                a = g_c.etypes# src: attn_score
                for tp in g_c.etypes:
                    src, dst = g_c.edges(etype=tp)
                    for j, (ss, dd) in enumerate(zip(src.tolist(), dst.tolist())):
                        if (dd in node_attn[l + 1]):  # connected to top layer nodes
                            node_attn[l][ss] = attn[l][tp][j] * node_attn[l + 1][dd]

            # sum up for each node
            ne_attn = {}
            new2old = {}
            for j, n in enumerate(neighbors.tolist()):
                ne_attn[n] = 0
                new2old[j] = n

        for l in range(self.num_layers - 1, -1, -1):  # from top
            for newn in node_attn[l]:
                ne_attn[new2old[newn]] += node_attn[l][newn]
        sorted_ne = sorted(zip(ne_attn.keys(), ne_attn.values()), key=lambda x: x[1], reverse=True)
        return sorted_ne


    def explanation_to_graph(self, explanation, subgraph, stkid, top_k=5, maskout=False):
        if not maskout:
            g_m_nodes = [i[0] for i in explanation[:top_k]]
        else:
            g_m_nodes = [i[0] for i in explanation[top_k:]]
            top_k = 4
        if not stkid in g_m_nodes:
            g_m_nodes.append(stkid)
        g_m = dgl.node_subgraph(subgraph, g_m_nodes)
        new_stkid = (g_m.ndata['_ID'].tolist()).index(stkid)
        return g_m, new_stkid

