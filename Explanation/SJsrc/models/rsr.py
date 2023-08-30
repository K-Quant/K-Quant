# This is a dgl implementation of RSR: Temporal Relational Ranking for Stock Prediction
import torch
import torch.nn as nn
import dgl.function as fn
import dgl

from .graph_model import HeterographModel
from dgl.nn.pytorch import edge_softmax


class RSRConv(nn.Module):
    def __init__(self, in_feats, num_etypes):
        super().__init__()
        self._in_feats = in_feats
        self.head_weight = nn.Linear(in_feats, 1)
        self.tail_weight = nn.Linear(in_feats, 1)
        self.rel_weight = nn.Parameter(torch.FloatTensor(size=(num_etypes,1)), requires_grad=True)
        self._allow_zero_in_degree = True
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.head_weight.weight, gain=gain)
        nn.init.xavier_normal_(self.tail_weight.weight, gain=gain)
        nn.init.xavier_normal_(self.rel_weight, gain=gain)

    def forward(self, graph, nfeat, get_attention=False):
        with graph.local_scope():
            funcs = {}
            graph.ndata['ft'] = nfeat
            graph.ndata['el'] = self.head_weight(nfeat)
            graph.ndata['er'] = self.tail_weight(nfeat)

            for _, etype, _ in graph.canonical_etypes:
                graph.apply_edges(fn.u_add_v("el", "er", "a"), etype=etype)
                graph.edges[etype].data["a"] += self.rel_weight[int(etype)]

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = edge_softmax(hg, hg.edata.pop("a"))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"),fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            attn = graph.edata.pop("a")
            attn = {k[1]: attn[k] for k in attn}

            if get_attention:
                return  graph.ndata.pop('ft'), attn
            return graph.ndata.pop('ft')


class RSRModel(HeterographModel):
    '''
        To implement a model, you need to specify the conv_layers and define forward_graph().
        You may also want to specify fc_out() depending on the graph output.
    '''
    def __init__(self, d_feat, hidden_size, num_etypes, num_layers, dropout, base_model, use_residual=False):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, base_model=base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.use_residual = use_residual
        if use_residual:
            self.fc_out = nn.Linear(hidden_size*2, 1)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)
        self.conv_layer = RSRConv(hidden_size, num_etypes)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_out.weight, gain=gain)

    def get_attention(self, graph):
        h = graph.ndata['nfeat']
        h, layer_attention = self.conv_layer(graph, h, get_attention=True) # [E,*,H,1]
        return [layer_attention]

    def forward_graph(self, h, index=None, return_subgraph=False):
        if index:
            subgraph = dgl.node_subgraph(self.g, index)
        else:
            subgraph = self.g

        h = self.conv_layer(subgraph, h)
        h = h.flatten(1)
        if return_subgraph:
            return h, subgraph
        else:
            return h




