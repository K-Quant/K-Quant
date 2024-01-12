from __future__ import division
from __future__ import print_function

import copy

import torch
import dgl
import numpy as np
import random
from .base import GraphExplainer


class xPath(GraphExplainer):
    '''
    This implementation treats a path as a sequence of nodes since there can be multiple paths between a pair of nodes.
    '''
    def __init__(self, graph_model, num_layers, device):
        super().__init__(graph_model, num_layers, device)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        self.one_hop_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        self.target_ntype = 'stock'
        self.random_seed = 2023
        self.scale = 100
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        dgl.seed(self.random_seed)

    def sample_step(self, g, ap, sample_n):
        apid = list(ap)

        one_hop_loader = dgl.dataloading.DataLoader(g, torch.tensor([apid[-1]], dtype=torch.int64, device=self.device),
                                                        self.one_hop_sampler, batch_size=1, shuffle=False,
                                                        drop_last=False)
        neighbors = None
        for neighbors, _, _ in one_hop_loader:
            break
        res = neighbors.detach().cpu().tolist()
        #print('num of neighbors:', len(res))
        if len(res) > sample_n:
            res = random.sample(res, sample_n)
        for aid in apid:
            if aid in res:
                res.remove(aid)
        path = [apid + [i] for i in res]
        return path

    def get_proxy_graph(self, g, p):
        g = g.to('cpu')

        if len(p) == 1:
            return

        x = g.ndata["nfeat"].clone().detach().cpu()
        num_stock = g.num_nodes(self.target_ntype)
        proxy_ids = [-1]
        # make nodes
        for i in range(1, len(p) - 1):
            node_id = p[i]
            node_feature = x[node_id, :].reshape(shape=(1, -1))

            proxy_ids.append(num_stock)
            #x = torch.concat([x, node_feature], dim=0)
            x = torch.cat([x, node_feature], dim=0)
            num_stock += 1

        sg_edges = {}
        for stp, etp, ttp in g.canonical_etypes:
            sg_edges[(stp, etp, ttp)] = [g.edges(etype=etp)[0].tolist(), g.edges(etype=etp)[1].tolist()]

        new_edges = {k: [[], []] for k in g.canonical_etypes}
        for etp in g.canonical_etypes:
            del_id = -1
            for i in range(1, len(p)):
                for j in range(len(sg_edges[etp][0])):
                    sid = sg_edges[etp][0][j]
                    tid = sg_edges[etp][1][j]
                    if sid == p[i]:
                        if i<len(p)-1 and (tid != p[i - 1]) and (tid != p[i]) and (tid != p[i + 1]): # not along the path
                            new_edges[etp][0] += [proxy_ids[i]]
                            new_edges[etp][1] += [tid]
                        elif tid == p[i - 1]: # along
                            if i > 1 and i<len(p)-1:
                                new_edges[etp][0] += [proxy_ids[i]]
                                new_edges[etp][1] += [proxy_ids[i-1]]
                            elif i==len(p)-1:
                                new_edges[etp][0] += [sid]
                                new_edges[etp][1] += [proxy_ids[i-1]]
                                del_id = j
                        elif i<len(p)-1 and tid == p[i]: # self loop
                            new_edges[etp][0] += [proxy_ids[i]]
                            new_edges[etp][1] += [proxy_ids[i]]
                        elif i<len(p)-1 and tid == p[i+1]: # reverse
                            new_edges[etp][0] += [proxy_ids[i]]
                            if i == len(p)-2:
                                new_edges[etp][1] += [tid]
                            else:
                                new_edges[etp][1] += [proxy_ids[i+1]]
            if del_id>=0: # delete the edges in the first step of the path
                #print(f'deleting {sg_edges[etp][0][del_id]}->{sg_edges[etp][1][del_id]}')
                sg_edges[etp][0].pop(del_id)
                sg_edges[etp][1].pop(del_id)
            #print(f'adding {new_edges[etp][0]}->{new_edges[etp][1]}')
            # sg_edges[etp][0] += new_edges[etp][0]
            # sg_edges[etp][1] += new_edges[etp][1]
            sg_edges[etp] = (sg_edges[etp][0], sg_edges[etp][1])

        #print('proxy graph edges:', sg_edges)
        sg = dgl.heterograph(sg_edges)
        sg.ndata['nfeat'] = x
        return sg.to(self.device)

    def get_proxy_homograph(self, g, p):
        g = g.to('cpu')

        if len(p) == 1:
            return

        x = g.ndata["nfeat"].clone().detach().cpu()
        num_stock = g.num_nodes()
        proxy_ids = [-1]
        # make nodes
        for i in range(1, len(p) - 1):
            node_id = p[i]
            node_feature = x[node_id, :].reshape(shape=(1, -1))

            proxy_ids.append(num_stock)
            #x = torch.concat([x, node_feature], dim=0)
            x = torch.cat([x, node_feature], dim=0)
            num_stock += 1

        sg_edges = [g.edges()[0].tolist(), g.edges()[1].tolist()]

        new_edges = [[],[]]

        del_id = -1
        for i in range(1, len(p)):
            for j in range(len(sg_edges[0])):
                sid = sg_edges[0][j]
                tid = sg_edges[1][j]
                if sid == p[i]:
                    if i<len(p)-1 and (tid != p[i - 1]) and (tid != p[i]) and (tid != p[i + 1]): # not along the path
                        new_edges[0] += [proxy_ids[i]]
                        new_edges[1] += [tid]
                    elif tid == p[i - 1]: # along
                        if i > 1 and i<len(p)-1:
                            new_edges[0] += [proxy_ids[i]]
                            new_edges[1] += [proxy_ids[i-1]]
                        elif i==len(p)-1:
                            new_edges[0] += [sid]
                            new_edges[1] += [proxy_ids[i-1]]
                            del_id = j
                    elif i<len(p)-1 and tid == p[i]: # self loop
                        new_edges[0] += [proxy_ids[i]]
                        new_edges[1] += [proxy_ids[i]]
                    elif i<len(p)-1 and tid == p[i+1]: # reverse
                        new_edges[0] += [proxy_ids[i]]
                        if i == len(p)-2:
                            new_edges[1] += [tid]
                        else:
                            new_edges[1] += [proxy_ids[i+1]]
        if del_id>=0:
            sg_edges[0].pop(del_id)
            sg_edges[1].pop(del_id)
            #print(f'adding {new_edges[etp][0]}->{new_edges[etp][1]}')
        # sg_edges[0] += new_edges[0]
        # sg_edges[1] += new_edges[1]
        sg_edges = (sg_edges[0], sg_edges[1])

        sg = dgl.graph(sg_edges)
        sg.ndata['nfeat'] = x
        sg.ndata[dgl.NID] = g.ndata[dgl.NID]
        return sg.to(self.device)

    def dense2dgl(self, rel_matrix, feature, device):
        # convert adj matrix to dgl sparse graph
        if len(rel_matrix.shape) == 3:
            rel_matrix = rel_matrix.sum(axis=-1)  # [N, N]
        idx = rel_matrix.nonzero().t()
        dgl_graph = dgl.graph((idx[0], idx[1])).to(device)
        dgl_graph.ndata['nfeat'] = torch.Tensor(feature).to(device)
        return dgl_graph

    def dgl2dense(self, rel_matrix, g):
        g_adj = torch.zeros(g.num_nodes(), g.num_nodes(), rel_matrix.shape[-1])
        dst, src = g.edges()
        g_nids = g.ndata[dgl.NID]
        g_adj[src, dst, :] = rel_matrix[g_nids[src], g_nids[dst], :]
        return g_adj

    def explain(self, full_model, graph, stkid, beam=5, sample_n=10):
        dataloader = dgl.dataloading.DataLoader(graph,
                                                        torch.Tensor([stkid]).type(torch.int64).to(self.device),
                                                        self.sampler,
                                                        batch_size=1, shuffle=False, drop_last=False)

        neighbors = None
        for neighbors, _, _ in dataloader:
            break

        target_id = stkid
        g_c = dgl.node_subgraph(graph, neighbors)  # induce the computation graph

        origin_ids = g_c.ndata['_ID'].tolist()
        new_target_id = origin_ids.index(target_id)
        g_c = g_c.to(self.device)
        origin_pred = full_model.predict_on_graph(g_c).detach().cpu().numpy()[new_target_id]*10000

        ancestor_p = [(new_target_id,)]
        top_k_p = {(new_target_id,): -100}
        visited = {}
        path2s = {}
        while len(ancestor_p) > 0:
            for ap in ancestor_p:
                paths = self.sample_step(g_c, ap, sample_n=sample_n)
                for pid in range(len(paths)):
                    p = paths[pid]
                    path_key = tuple(p)
                    if self.graph_model == 'heterograph':
                        shadow_graph = self.get_proxy_graph(g_c, p)
                    else:
                        shadow_graph = self.get_proxy_homograph(g_c, p)
                    pred = full_model.predict_on_graph(shadow_graph).detach().cpu().numpy()[new_target_id]*10000
                    tmp = abs(origin_pred-pred)
                    path2s[path_key] = tmp
                    top_k_p[path_key] = tmp

            values = list(top_k_p.values())
            keys = list(top_k_p.keys())
            ind = np.argsort(values)[-beam:]
            top_k_p = {keys[b]: top_k_p[keys[b]] for b in ind}

            ancestor_p = []
            for b in top_k_p:
                if (len(b) <= self.num_layers) and (not b in visited):
                    ancestor_p.append(b)
                    visited[b] = 1

        xpath2s = {}
        for path_key in path2s:
            origin_path_key = [origin_ids[i]for i in path_key]
            xpath2s[tuple(origin_path_key)] = path2s[path_key]
        return xpath2s

    def explain_dense(self, full_model, original_pred, g, rel_matrix, stock_id,
                      beam=5, sample_n=10, get_fidelity=False, top_k=5):
        """explain_dense is a newer function that support models trained with dense adjacency matrix"""
        target_id = stock_id
        neighbors = rel_matrix[:, target_id, :].sum(axis=-1).nonzero().squeeze().tolist()
        neighbors.append(target_id)

        # g = self.dense2dgl(rel_matrix, feature, self.device)
        g_c = dgl.node_subgraph(g, neighbors)  # induce the computation graph

        origin_ids = g_c.ndata['_ID'].tolist()
        new_target_id = origin_ids.index(target_id)
        # original_pred = full_model(g_c.ndata['nfeat'], self.dgl2dense(rel_matrix, g_c)).detach().cpu().numpy()[new_target_id]
        original_pred = original_pred[stock_id]

        ancestor_p = [(new_target_id,)]
        top_k_p = {(new_target_id,): -100}
        visited = {}
        path2s = {}
        while len(ancestor_p) > 0:
            for ap in ancestor_p:
                paths = self.sample_step(g_c, ap, sample_n=sample_n)
                for pid in range(len(paths)):
                    p = paths[pid]
                    path_key = tuple(p)
                    shadow_graph = self.get_proxy_homograph(g_c, p)
                    pred = full_model(shadow_graph.ndata['nfeat'], self.dgl2dense(rel_matrix, shadow_graph)).detach().cpu().numpy()[new_target_id]
                    tmp = abs(original_pred * self.scale - pred * self.scale)
                    path2s[path_key] = tmp
                    top_k_p[path_key] = tmp

            values = list(top_k_p.values())
            keys = list(top_k_p.keys())
            ind = np.argsort(values)[-beam:]
            top_k_p = {keys[b]: top_k_p[keys[b]] for b in ind}

            ancestor_p = []
            for b in top_k_p:
                if (len(b) <= self.num_layers) and (not b in visited):
                    ancestor_p.append(b)
                    visited[b] = 1

        # normalize the score of path2s
        scores = np.array(list(path2s.values()))
        if np.sum(scores) == 0:
            scores = np.ones_like(scores)
        else:
            scores = scores / np.sum(scores)
        for i in range(len(scores)):
            path2s[list(path2s.keys())[i]] = int(scores[i] * 100) / 100

        # convert to original ids in g
        xpath2s = {}
        for path_key in path2s:
            origin_path_key = origin_ids[path_key[1]]
            xpath2s[origin_path_key] = path2s[path_key]

        # get top k explanations
        xpath_nodes = list(xpath2s.keys())
        if top_k >= len(xpath_nodes):
            exp_nodes = xpath_nodes
            explanation = xpath2s
        else:
            ind = np.argsort(list(xpath2s.values()))[-top_k:]
            exp_nodes = [xpath_nodes[i] for i in ind]
            explanation = {k: xpath2s[k] for k in exp_nodes}

        if get_fidelity:
            if target_id not in exp_nodes:
                exp_nodes.append(target_id)
            g_m = dgl.node_subgraph(g, exp_nodes)
            g_m_target_id = g_m.ndata[dgl.NID].tolist().index(target_id)
            g_m_pred = full_model(g_m.ndata['nfeat'], self.dgl2dense(rel_matrix, g_m)).detach().cpu().numpy()[g_m_target_id]
            fidelity = abs(original_pred - g_m_pred) / self.scale
            return explanation, fidelity

        return explanation

    def explanation_to_graph(self, explanation, subgraph, stkid, top_k=5, maskout=False):
        keys = list(explanation.keys())
        values = list(explanation.values())
        if not maskout:
            ind = np.argsort(values)[-top_k:]
        else:
            ind = np.argsort(values)[:-top_k]
        paths = [keys[i] for i in ind]
        xnodes = {}
        nodepair = {}
        #print(paths)
        for path in paths:
            for i in range(len(path)):
                if i>0: # along
                    nodepair[(path[i], path[i-1])] = 1
                nodepair[(path[i], path[i])] = 1
                if i<len(path)-1: # reverse
                    nodepair[(path[i], path[i+1])] = 1
                xnodes[path[i]] = 1
        if not stkid in xnodes: # for those with no graph structure
            xnodes[stkid] = 1
            nodepair[(stkid, stkid)] = 1

        path_graph = dgl.node_subgraph(subgraph, list(xnodes.keys()))
        origin_ids = path_graph.ndata['_ID'].tolist()
        if self.graph_model=='heterograph':
            sg_edges = {}
            for etp in path_graph.canonical_etypes:
                sg_edges[etp] = [path_graph.edges(etype=etp[1])[0].tolist(), path_graph.edges(etype=etp[1])[1].tolist()]
            #print(origin_ids)
            # check every edge to be in some path
            for etp in path_graph.canonical_etypes:
                del_id = []
                for j in range(len(sg_edges[etp][0])):

                    sid = origin_ids[sg_edges[etp][0][j]]
                    tid = origin_ids[sg_edges[etp][1][j]]
                    #print(etp, sid, tid)
                    if not (sid, tid) in nodepair:
                        del_id.append(j)
                        #print(f'delete {j}')

                if len(del_id)>0:  # delete the edges in the first step of the path
                    del_id.reverse()
                    for i in del_id:
                        sg_edges[etp][0].pop(i)
                        sg_edges[etp][1].pop(i)
            #print('xgraph edges:', sg_edges)
            new_stkid = origin_ids.index(stkid)
            for etp in sg_edges:
                sg_edges[etp] = (sg_edges[etp][0], sg_edges[etp][1])
            g_m = dgl.heterograph(sg_edges).to(self.device)
        else:
            sg_edges = [path_graph.edges()[0].tolist(), path_graph.edges()[1].tolist()]
            del_id = []
            for j in range(len(sg_edges[0])):
                sid = origin_ids[sg_edges[0][j]]
                tid = origin_ids[sg_edges[1][j]]
                if not (sid, tid) in nodepair:
                    del_id.append(j)
            if len(del_id) > 0:  # delete the edges in the first step of the path
                del_id.reverse()
                for i in del_id:
                    sg_edges[0].pop(i)
                    sg_edges[1].pop(i)
            new_stkid = origin_ids.index(stkid)
            sg_edges = (sg_edges[0], sg_edges[1])
            g_m = dgl.graph(sg_edges).to(self.device)
        g_m.ndata['nfeat'] = path_graph.ndata['nfeat']
        return g_m, new_stkid
