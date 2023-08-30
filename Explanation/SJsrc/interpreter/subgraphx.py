from __future__ import division
from __future__ import print_function

import torch
import dgl
from .base import GraphExplainer
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn


class MCTSNode:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        return str(self.nodes)


class SubgraphX(nn.Module):
    def __init__(
            self,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=3,
            node_min=4,
            shapley_steps=10,
            log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = None

    def set_model(self, model):
        self.model = model

    def shapley(self, subgraph_nodes):
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                g = self.graph
                with g.local_scope():
                    g.ndata['nfeat'] = exclude_feat
                    exclude_value = self.model.predict_on_graph(g)
                    g.ndata['nfeat'] = include_feat
                    include_value = self.model.predict_on_graph(g)
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = dgl.node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = dgl.remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = dgl.to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[dgl.NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[dgl.NID][largest_cc_nids].sort().values
            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
                          + self.coef
                          * c.immediate_reward
                          * children_visit_sum_sqrt
                          / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, **kwargs):

        self.model.eval()
        min_nodes = self.node_min if graph.num_nodes() > self.node_min else graph.num_nodes()

        self.graph = graph
        self.feat = graph.ndata["nfeat"]
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root = MCTSNode(graph.nodes())
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        return self.mcts_node_maps

        # best_leaf = None
        # best_immediate_reward = float("-inf")
        # for mcts_node in self.mcts_node_maps.values():
        #     if len(mcts_node.nodes) > min_nodes:
        #         continue

        #     if mcts_node.immediate_reward > best_immediate_reward:
        #         best_leaf = mcts_node
        #         best_immediate_reward = best_leaf.immediate_reward

        # return best_leaf.nodes, best_immediate_reward


class HeteroSubgraphX(nn.Module):
    def __init__(
            self,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=3,
            node_min=4,
            shapley_steps=10,
            log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = None
        self.target_ntype = None

    def set_model(self, model):
        self.model = model

    def shapley(self, subgraph_nodes):
        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_regions = {
            ntype: nodes.tolist() for ntype, nodes in subgraph_nodes.items()
        }
        for _ in range(self.num_hops - 1):
            for c_etype in self.graph.canonical_etypes:
                src_ntype, _, dst_ntype = c_etype
                if (
                        src_ntype not in local_regions
                        or dst_ntype not in local_regions
                ):
                    continue

                in_neighbors, _ = self.graph.in_edges(
                    local_regions[dst_ntype], etype=c_etype
                )
                _, out_neighbors = self.graph.out_edges(
                    local_regions[src_ntype], etype=c_etype
                )
                local_regions[src_ntype] = list(
                    set(local_regions[src_ntype] + in_neighbors.tolist())
                )
                local_regions[dst_ntype] = list(
                    set(local_regions[dst_ntype] + out_neighbors.tolist())
                )

        split_point = self.graph.num_nodes()
        coalition_space = {
            ntype: list(
                set(local_regions[ntype]) - set(subgraph_nodes[ntype].tolist())
            )
                   + [split_point]
            for ntype in subgraph_nodes.keys()
        }

        marginal_contributions = []
        for _ in range(self.shapley_steps):
            selected_node_map = dict()
            for ntype, nodes in coalition_space.items():
                permuted_space = np.random.permutation(nodes)
                split_idx = int(np.where(permuted_space == split_point)[0])
                selected_node_map[ntype] = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = {
                ntype: torch.ones(self.graph.num_nodes(ntype))
                for ntype in self.graph.ntypes
            }
            for ntype, region in local_regions.items():
                exclude_mask[ntype][region] = 0.0
            for ntype, selected_nodes in selected_node_map.items():
                exclude_mask[ntype][selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = {
                self.target_ntype: exclude_mask[ntype].clone()
            }
            for ntype, subgn in subgraph_nodes.items():
                exclude_mask[ntype][subgn] = 1.0

            exclude_feat = self.feat * exclude_mask[self.target_ntype].unsqueeze(1).to(self.feat.device)

            include_feat = self.feat * include_mask[self.target_ntype].unsqueeze(1).to(self.feat.device)

            with torch.no_grad():
                g = self.graph
                with g.local_scope():
                    g.ndata['nfeat'] = exclude_feat
                    exclude_value = self.model.predict_on_graph(g)
                    g.ndata['nfeat'] = include_feat
                    include_value = self.model.predict_on_graph(g)

            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = dgl.node_subgraph(self.graph, mcts_node.nodes)
        # Choose k nodes based on the highest degree in the subgraph
        node_degrees_map = {
            ntype: torch.zeros(
                subg.num_nodes(ntype), device=subg.nodes(ntype).device
            )
            for ntype in subg.ntypes
        }
        for c_etype in subg.canonical_etypes:
            src_ntype, _, dst_ntype = c_etype
            node_degrees_map[src_ntype] += subg.out_degrees(etype=c_etype)
            node_degrees_map[dst_ntype] += subg.in_degrees(etype=c_etype)

        node_degrees_list = [
            ((ntype, i), degree)
            for ntype, node_degrees in node_degrees_map.items()
            for i, degree in enumerate(node_degrees)
        ]
        node_degrees = torch.stack([v for _, v in node_degrees_list])
        k = min(subg.num_nodes(), self.num_child)
        chosen_node_indicies = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices
        chosen_nodes = [node_degrees_list[i][0] for i in chosen_node_indicies]

        mcts_children_maps = dict()

        for ntype, node in chosen_nodes:
            new_subg = dgl.remove_nodes(subg, node, ntype, store_ids=True)

            if new_subg.num_edges() > 0:
                new_subg_homo = dgl.to_homogeneous(new_subg)
                # Get the largest weakly connected component in the subgraph.
                nx_graph = dgl.to_networkx(new_subg_homo.cpu())
                largest_cc_nids = list(
                    max(nx.weakly_connected_components(nx_graph), key=len)
                )
                largest_cc_homo = dgl.node_subgraph(new_subg_homo, largest_cc_nids)
                largest_cc_hetero = dgl.to_heterogeneous(
                    largest_cc_homo, new_subg.ntypes, new_subg.etypes
                )

                # Follow steps for backtracking to original graph node ids
                # 1. retrieve instanced homograph from connected-component homograph
                # 2. retrieve instanced heterograph from instanced homograph
                # 3. retrieve hetero-subgraph from instanced heterograph
                # 4. retrieve orignal graph ids from subgraph node ids
                indicies = largest_cc_hetero.ndata[dgl.NID]
                cc_nodes = {
                    self.target_ntype: subg.ndata[dgl.NID][
                        new_subg.ndata[dgl.NID][
                            new_subg_homo.ndata[dgl.NID][
                                largest_cc_homo.ndata[dgl.NID][indicies]
                            ]
                        ]
                    ]
                }
            else:
                available_ntypes = [
                    ntype
                    for ntype in new_subg.ntypes
                    if new_subg.num_nodes(ntype) > 0
                ]
                chosen_ntype = np.random.choice(available_ntypes)
                # backtrack from subgraph node ids to entire graph
                chosen_node = subg.ndata[dgl.NID][chosen_ntype][
                    np.random.choice(new_subg.nodes[chosen_ntype].data[dgl.NID])
                ]
                cc_nodes = {
                    chosen_ntype: torch.tensor(
                        [chosen_node],
                        device=subg.device,
                    )
                }

            if str(cc_nodes) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(cc_nodes)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(cc_nodes)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        r"""Perform a MCTS rollout.

        Parameters
        ----------
        mcts_node : MCTSNode
            Starting node for MCTS

        Returns
        -------
        float
            Reward for visiting the node this time
        """
        if (
                sum(len(nodes) for nodes in mcts_node.nodes.values())
                <= self.node_min
        ):
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
                          + self.coef
                          * c.immediate_reward
                          * children_visit_sum_sqrt
                          / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, **kwargs):
        self.model.eval()
        min_nodes = self.node_min if graph.num_nodes() > self.node_min else graph.num_nodes()
        # assert (
        #     graph.num_nodes() > self.node_min
        # ), f"The number of nodes in the\
        #     graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = graph.ndata["nfeat"]
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root_dict = {ntype: graph.nodes(ntype) for ntype in graph.ntypes}
        root = MCTSNode(root_dict)
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        return self.mcts_node_maps
        # best_leaf = None
        # best_immediate_reward = float("-inf")
        # for mcts_node in self.mcts_node_maps.values():
        #     if len(mcts_node.nodes[self.target_ntype]) > min_nodes:
        #         continue

        #     if mcts_node.immediate_reward > best_immediate_reward:
        #         best_leaf = mcts_node
        #         best_immediate_reward = best_leaf.immediate_reward

        # return best_leaf.nodes, best_immediate_reward


class SubgraphXExplainer(GraphExplainer):
    def __init__(self, graph_model, num_layers, device):
        super().__init__(graph_model, num_layers, device)
        self.subgraphx = HeteroSubgraphX(num_hops=num_layers) if graph_model == 'heterograph' else SubgraphX(
            num_hops=num_layers)
        self.target_ntype = 's'
        if graph_model == 'heterograph':
            self.subgraphx.target_ntype = self.target_ntype
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)

    def explain(self, full_model, graph, stkid):
        self.subgraphx.set_model(full_model)
        target_ntype = self.target_ntype
        dataloader = dgl.dataloading.DataLoader(graph,
                                                torch.Tensor([stkid]).type(torch.int64).to(self.device),
                                                self.sampler,
                                                batch_size=1, shuffle=False, drop_last=False)

        neighbors = None
        for neighbors, _, _ in dataloader:
            break

        target_id = stkid
        g_c = dgl.node_subgraph(graph, neighbors)  # induce the computation graph
        mcts_node_maps = self.subgraphx.explain_graph(g_c)

        g_nodes_explain = []
        for mcts_node in mcts_node_maps.values():
            if self.graph_model == 'heterograph':
                g_nodes_explain.append([g_c.ndata['_ID'][mcts_node.nodes[target_ntype]].detach().cpu().numpy().tolist(),
                                        mcts_node.immediate_reward])
            else:
                g_nodes_explain.append([g_c.ndata['_ID'][mcts_node.nodes].detach().cpu().numpy().tolist(),
                                        mcts_node.immediate_reward])

        # if self.graph_model == 'heterograph':
        #     g_nodes_explain_ne = g_c.ndata['_ID'][g_nodes_explain[target_ntype]]
        # else:
        #     g_nodes_explain_ne = g_c.ndata['_ID'][g_nodes_explain]
        # explanation = {'subgraph nodes': g_nodes_explain_ne.detach().cpu().numpy().tolist(), 'reward': reward}
        return g_nodes_explain

    def explanation_to_graph(self, explanation, subgraph, stkid, top_k=5, maskout=False):
        best_immediate_reward = float("-inf")
        best_exp = None
        min_len = float("inf")
        for exp in explanation:
            if len(exp[0]) < min_len:
                min_len = len(exp[0])
        if min_len > top_k:
            top_k = min_len
        for exp in explanation:
            if len(exp[0]) > top_k:
                continue
            if exp[1] > best_immediate_reward:
                best_immediate_reward = exp[1]
                best_exp = exp[0]

        if not maskout:
            g_m_nodes = best_exp
        else:
            if self.graph_model == 'heterograph':
                g_nodes = subgraph.nodes(ntype=self.target_ntype).detach().cpu().numpy().tolist()
            else:
                g_nodes = subgraph.nodes().detach().cpu().numpy().tolist()
            g_m_nodes = list(set(g_nodes) - set(best_exp))

        if not stkid in g_m_nodes:
            g_m_nodes.append(stkid)
        g_m = dgl.node_subgraph(subgraph, g_m_nodes)
        new_stkid = (g_m.ndata['_ID'].tolist()).index(stkid)
        return g_m, new_stkid
