import torch
import dgl
import copy
import random
import numpy as np
import torch.nn as nn
import pandas as pd
import networkx as nx
from pgmpy.estimators.CITests import g_sq


class HencexExplainer():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_type = 's' # stock
        self.rel_matrix = None
        random.seed(2024)
        np.random.seed(2024)

    # def dense2sparse(self, rel_matrix, feature, device):
    #     num_rels = rel_matrix.shape[-1]
    #     edge_index = {}
    #     for i in range(num_rels):
    #         idx = rel_matrix[:, :, i].nonzero()
    #         if idx.size(0) > 0:
    #             edge_index[(self.target_type, i, self.target_type)] = (idx[0], idx[1])
    #     g = dgl.heterograph(edge_index)
    #     g.nodes[self.target_type].data['feat'] = feature
    #     return g.to(device)

    def dense2sparse(self, rel_matrix, feature, device):
        # convert adj matrix to dgl sparse graph
        if len(rel_matrix.shape) == 3:
            rel_matrix = rel_matrix.sum(axis=-1)  # [N, N]
        unit_matrix_2d = torch.eye(300)
        rel_matrix = rel_matrix + unit_matrix_2d
        idx = rel_matrix.nonzero(as_tuple=True)
        dgl_graph = dgl.graph((idx[0], idx[1])).to(device)
        dgl_graph.ndata['feat'] = torch.Tensor(feature).to(device)
        return dgl_graph

    # def sparse2dense(self, rel_matrix, g):
    #     g_adj = torch.zeros(g.num_nodes(), g.num_nodes(), rel_matrix.shape[-1])
    #     for etp in g.canonical_etypes:
    #         src, dst = g.edges(form='uv', etype=etp)
    #         g_nids = g.nodes(self.target_type).data[dgl.NID]
    #         g_adj[src, dst, int(etp[1])] = rel_matrix[g_nids[src], g_nids[dst], int(etp[1])]
    #     return g_adj

    def sparse2dense(self, rel_matrix, g):
        g_adj = torch.zeros(g.num_nodes(), g.num_nodes(), rel_matrix.shape[-1])
        src, dst = g.edges()
        g_nids = g.ndata[dgl.NID]
        g_adj[src, dst, :] = rel_matrix[g_nids[src], g_nids[dst], :]
        return g_adj

    def adjustPperturb(self, val):
        p_rate = 0
        for node in self.sampled_data.keys():
            p_rate += np.count_nonzero(
                np.sum(self.sampled_data[node][(self.cat_y_cap!=val).nonzero()[0], :], axis=1))/np.count_nonzero(self.cat_y_cap!=val)
        return p_rate / len(self.sampled_data)

    def uniformPerturb(self, new_target_id, g_c, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01):
        num_RV = n_cat_value
        node_feature = g_c.ndata['feat']
        # nid_mapping = g_c.nodes[self.target_type].data[dgl.NID]
        for n in range(g_c.num_nodes()):
            num_RV += node_feature[n].count_nonzero().item()
        # num_samples = max(k*num_RV, num_samples)
        self.sampled_data = {}
        sampled_y_cap = []
        for i in range(num_samples):
            X_perturb = copy.deepcopy(node_feature)
            for node in range(g_c.num_nodes()):
                seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]
                pos = node_feature[node].nonzero().cpu().numpy()
                if seed == 1:
                    perturb_array = torch.tensor(np.random.choice(2, size=pos.shape[0], p=[0.5, 0.5]),
                                                 dtype=torch.float32).to(self.device)
                    tmp = (X_perturb[node][pos.T] != perturb_array).detach().cpu().numpy()
                    tmp = np.expand_dims(tmp, axis=0)
                    X_perturb[node][pos.T] = perturb_array
                else:
                    tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)
                if node not in self.sampled_data:
                    self.sampled_data[node] = tmp
                else:
                    self.sampled_data[node] = np.append(self.sampled_data[node], tmp, axis=0)
            with torch.no_grad():
                g_c_adj = self.sparse2dense(self.rel_matrix, g_c)
                out = self.model(X_perturb, g_c_adj)
            pred_score = out[new_target_id].item()
            sampled_y_cap.append(pred_score)
        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T
        perturb_range = self.ori_sampled_y_cap.max() - self.ori_sampled_y_cap.min()

        if perturb_range==0:
            # print('GNN prediction never change!')
            self.cat_y_cap = np.ones(self.ori_sampled_y_cap.shape)
            return
        elif perturb_range<pred_threshold:
            # print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.cat_y_cap = np.where(self.ori_sampled_y_cap <= (self.ori_sampled_y_cap.min() + pred_threshold), 0,
                                  self.ori_sampled_y_cap)
        self.cat_y_cap = np.where(self.cat_y_cap >= (self.ori_sampled_y_cap.max() - pred_threshold), 2, self.cat_y_cap)
        self.cat_y_cap = np.where((0 < self.cat_y_cap) & (self.cat_y_cap < 2), 1, self.cat_y_cap)
        self.cat_y_cap = self.cat_y_cap.astype(int)
        # counts = np.array([np.count_nonzero(self.cat_y_cap == val) for val in range(3)])
        # bar = 0.001
        # how_many_more = np.where(bar * num_samples - counts < 0, 0, np.ceil(bar * num_samples - counts)).astype(int)
        # print(how_many_more)

    def basketSearching(self, new_target_id, g_c, p_threshold):
        basket = {}
        g_c = g_c.to_networkx()
        target = new_target_id
        U = set([target])
        processed = set()
        c = 0
        while len(U) > 0:
            c += 1
            current_U = U.copy()
            for u in current_U:
                S_data = np.concatenate((self.cat_y_cap.astype(np.int8), self.sampled_data[u]), axis=1)
                pdData = pd.DataFrame(S_data)
                ind_ori_to_sub = dict(zip(['target'] + ['f' + str(i) for i in range(self.sampled_data[u].shape[1])],
                                          list(pdData.columns)))
                feat_p_values = [
                    g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for
                    feat_ in ['f' + str(i) for i in range(self.sampled_data[u].shape[1])]]
                # a lower p_threshold will lead to basket = {} for some nodes
                if u == target:
                    feats_to_pick = [i for i, x in enumerate(feat_p_values)]
                else:
                    feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]
                if len(feats_to_pick) > 0:
                    basket[u] = np.array(feats_to_pick)
                    U = U.union(set(g_c[u]))
                elif len(feats_to_pick) == 0 and u == target:
                    return {}
            processed = processed.union(current_U)
            U = U.difference(processed)
        return basket

    def blanketShrinking(self, target_, g, S, p_threshold):
        g = g.to_networkx().to_undirected()

        pdData = self.cat_y_cap.astype(np.int8)

        can_remove = []
        for node in [n for n in S if n != target_]:
            subgraph = g.subgraph([n for n in S if n != node])
            if nx.is_connected(subgraph):
                can_remove.append(node)

        for node in S:
            pdData = np.concatenate((pdData, self.sampled_data[node]), axis=1)

        for node in S:
            pdData = np.concatenate((pdData, self.node_var[node]), axis=1)

        pdData = pd.DataFrame(pdData)
        ind_ori_to_sub = dict(zip(['target'] + S + ['syn' + str(n) for n in S], list(pdData.columns)))

        p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for
                    node in can_remove]

        selected_nodes = set(S)
        for idx in np.argsort(-1 * np.asarray(p_values)):  # trick to reduce conditional independency test, start from most independent variable (largest p value)
            node = can_remove[idx]
            if g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub['syn' + str(node)],
                    Z=[ind_ori_to_sub[n] for n in selected_nodes if n != node], data=pdData, boolean=True,
                    significance_level=p_threshold):
                selected_nodes.remove(node)

        return list(selected_nodes)

    def explainFeature(self, u, basket, S=[], p_threshold=.05):  # u is the node to explain

        # if self.dataset=='DBLP' and u%10 not in self.DBLP_idx2node_type:
        #     raise ValueError("DBLP illegal explain feature")

        if u not in self.sampled_data:
            raise ValueError("u not in sampled data")

        # print('Explaning features for node: ' + str(u))

        S_data = self.cat_y_cap.astype(np.int8)

        for n in S:
            # cat = self.vec2categ(self.sampled_data[n])  # trick to reduce number of columns
            S_data = np.concatenate((S_data, self.sampled_data[n]), axis=1)

        pdData = np.concatenate((S_data, self.sampled_data[u][:, basket]), axis=1)
        # print(pdData.shape)

        pdData = pd.DataFrame(pdData)

        ind_ori_to_sub = dict(zip(['target'] + S + ['f' + str(i) for i in basket], list(pdData.columns)))
        # print(ind_ori_to_sub.keys())

        # feat_p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]
        feat_p_values = [
            g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData,
                 boolean=False)[1] for feat_ in ['f' + str(i) for i in basket]]

        feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

        if len(feats_to_pick) == 0:
            # print('No feature is critical')
            return None, None

        else:
            # print(feats_to_pick)
            feats_to_pick.sort()
            return np.array(feats_to_pick), basket[np.array(feats_to_pick)]

    def explain(self, original_preds, g, rel_matrix, stock_id, top_k,
                 n_cat_value=3, num_samples=1000, k=1, p_perturb=0.5, p_threshold=0.3, pred_threshold=0.01):
        target_id = stock_id
        g_c = dgl.khop_in_subgraph(g, target_id, 1)[0].to(self.device)
        new_target_id = g_c.ndata['_ID'].tolist().index(target_id)
        self.rel_matrix = rel_matrix
        self.uniformPerturb(new_target_id, g_c, n_cat_value, num_samples, k, p_perturb, pred_threshold)
        blanket_basket = self.basketSearching(new_target_id, g_c, p_threshold)
        cand_nodes = list(blanket_basket.keys())
        if new_target_id not in cand_nodes:
            cand_nodes.insert(0, new_target_id)

        self.node_var = {node: self.sampled_data[node][:, blanket_basket[node]] for node in blanket_basket}

        S = []
        U = [new_target_id]
        I = set()
        raw_feature_exp = {}

        p_values = [-1]
        c = 0

        while len(U)>0 and min(p_values)<p_threshold:
            u = U[p_values.index(min(p_values))]
            I = I.union([U[i] for i in range(len(U)) if p_values[i]>=p_threshold])
            U = set(U)
            U.remove(u)

            feats_to_pick, raw_feature_exp[u] = self.explainFeature(u, blanket_basket[u], S=S, p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

            if raw_feature_exp[u] is None: # synthetic node overestimates feature importance
                raw_feature_exp.pop(u)
                I.add(u)

            else:
                self.sampled_data[u] = self.sampled_data[u][:, blanket_basket[u][feats_to_pick]]
                S.append(u)
                U = U.union(set(cand_nodes).difference(set(S)))

            U = U.difference(I)

            if len(U)>0: # only when U is not empty, compute p values

                U = list(U) # use list to fix order

                pdData = self.cat_y_cap.astype(np.int8)

                for node in S: # conditioned on the current E
                    pdData = np.concatenate((pdData, self.sampled_data[node]), axis=1)

                for node in U:
                    pdData = np.concatenate((pdData, self.node_var[node]), axis=1) # compute dependency of synthetic node variable

                pdData = pd.DataFrame(pdData)

                ind_ori_to_sub = dict(zip(['target'] + S + U, list(pdData.columns)))

                # p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
                p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in U]
                c+=1

        if len(S)==0:
            print('len(S)==0')
            S.append(new_target_id)

        if len(S)>1:
            S = self.blanketShrinking(new_target_id, g_c, list(S), p_threshold)
            feature_exp = {k: v for k, v in raw_feature_exp.items() if k in S}

        S.pop(S.index(new_target_id))
        explanation_nodes = list(S) if len(S) <= top_k else list(S)[:top_k]
        explanation = [g_c.ndata['_ID'][i].item() for i in explanation_nodes]

        return explanation
