from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from .graph_model import HeterographModel


class simpleHeteroGATConv(nn.Module):
    def __init__(
            self,
            edge_feats,
            num_etypes,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=True,
            bias=False,
            alpha=0.0,
    ):
        super(simpleHeteroGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats = self._in_dst_feats = in_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(num_etypes, edge_feats)))

        in_dim = None
        for name in in_feats:
            if in_dim:
                assert in_dim == in_feats[name]
            else:
                in_dim = in_feats[name]
        self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            if in_dim != num_heads * out_feats:
                self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)


    def forward(self, graph, nfeat, res_attn=None):
        with graph.local_scope():
            funcs = {}
            h = self.feat_drop(nfeat)
            feat = self.fc(h).view(-1, self._num_heads, self._out_feats)

            graph.ndata['ft'] = feat
            if self.res_fc is not None:
                graph.ndata['h'] = h

            for src, etype, dst in graph.canonical_etypes:
                feat_src = graph.nodes[src].data['ft']
                feat_dst = graph.nodes[dst].data['ft']
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.nodes[src].data['el'] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.nodes[dst].data['er'] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1).expand(graph.number_of_edges(etype),
                                                                             self._num_heads, 1)
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(graph.edges[etype].data.pop("e") + ee)

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[
                        etype] * self.alpha
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            rst = graph.ndata.pop('ft')
            if self.res_fc is not None:
                rst = self.res_fc(graph.ndata['h']).view(
                            graph.ndata['h'].shape[0], self._num_heads, self._out_feats) + rst
            if self.bias:
                rst = rst + self.bias_param

            if self.activation:
                rst = self.activation(rst)
            res_attn = {e: graph.edges[e].data["a"].detach() for e in graph.etypes}
            return rst, res_attn


class SimpleHeteroHGN(HeterographModel):
    '''
    To implement a model, you need to specify the conv_layers and define forward_graph().
    You may also want to specify fc_out() depending on the graph output.
    '''
    def __init__(self, d_feat, edge_dim, num_etypes, num_hidden, num_layers, dropout,
                 feat_drop,attn_drop,negative_slope,graph_layer_residual,alpha,base_model,num_graph_layer,heads=None, use_residual=False):
        super(SimpleHeteroHGN, self).__init__(
            base_model=base_model, d_feat=d_feat, hidden_size=num_hidden, num_layers=num_layers, dropout=dropout)

        self.num_layers = num_graph_layer
        self.conv_layers = nn.ModuleList()
        self.activation = F.elu

        if not heads: # set default attention heads
            heads = [1]*num_graph_layer

        #in_dims = {'none': num_hidden, self.target_type: num_hidden}
        in_dims = {self.target_type: num_hidden}
        self.conv_layers.append(
            simpleHeteroGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )

        for l in range(1, self.num_layers):
            in_dims = {n: num_hidden * heads[l - 1] for n in in_dims}
            self.conv_layers.append(
                simpleHeteroGATConv(
                    edge_dim,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    graph_layer_residual,
                    self.activation,
                    alpha=alpha,
                )
            )

        self.use_residual = use_residual
        if use_residual:
            self.fc_out = nn.Linear(num_hidden * heads[-1]+num_hidden, 1)
        else:
            self.fc_out = nn.Linear(num_hidden * heads[-1], 1)


    def get_attention(self, graph):
        h = graph.ndata['nfeat']
        attn = []
        res_attn = None
        for i, layer in enumerate(self.conv_layers):
            h, layer_attention = layer(graph, h, res_attn=res_attn) # [E,*,H,1]
            attn.append(layer_attention)
            h = h.flatten(1)
        return attn

    def forward_graph(self, h, index=None, return_subgraph=False):
        if index:
            subgraph = dgl.node_subgraph(self.g, {self.target_type: index})
        else:
            subgraph = self.g

        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.conv_layers[l](subgraph, h, res_attn=res_attn)
            h = h.flatten(1)
        if return_subgraph:
            return h, subgraph
        else:
            return h







