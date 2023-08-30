import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl
from .graph_model import HomographModel


class GATModel(HomographModel):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU", num_graph_layer=2, heads=None, use_residual=False):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, base_model=base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.use_residual = use_residual
        if use_residual:
            self.fc_out = nn.Linear(hidden_size*2, 1)
        else:
            self.fc_out = nn.Linear(hidden_size, 1)

        self.num_graph_layer = num_graph_layer
        self.gat_layers = nn.ModuleList()

        if not heads: # set default attention heads
            heads = [1]*num_graph_layer
        heads = [1] + heads

        for i in range(num_graph_layer-1):
            self.gat_layers.append(
                dglnn.GATConv(
                    hidden_size * heads[i],
                    hidden_size,
                    heads[i+1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                )
            )

        self.gat_layers.append(
            dglnn.GATConv(
                hidden_size * heads[-2],
                hidden_size,
                heads[-1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )
        self.reset_parameters()
        for layer in self.gat_layers:
            layer._allow_zero_in_degree = True
        self._allow_zero_in_degree = True

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_out.weight, gain=gain)

    def get_attention(self, graph):
        h = graph.ndata['nfeat']
        attn = []
        for i, layer in enumerate(self.gat_layers):
            h, layer_attention = layer(graph, h, get_attention=True) # [E,*,H,1]
            attn.append(layer_attention)
            if i == self.num_graph_layer-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return attn


    def forward(self, x, index = None):
        if not self.g:
            raise ValueError("graph not specified")
        h0 = self.forward_rnn(x)
        h = self.forward_graph(h0, index)
        return self.forward_predictor(h0, h)

    def forward_rnn(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return out[:, -1, :]

    def forward_graph(self, h, index=None, return_subgraph=False):
        if index:
            subgraph = dgl.node_subgraph(self.g, index)
        else:
            subgraph = self.g
        for i, layer in enumerate(self.gat_layers):
            h = layer(subgraph, h)
            if i == self.num_graph_layer-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        if return_subgraph:
            return h, subgraph
        else: return h


    def forward_predictor(self, h0, h):
        if self.use_residual:
            return self.fc_out(torch.cat([h0, h], dim=1)).reshape(shape=(-1,)) # you may need to use .concat() for some torch version
        else:
            return self.fc_out(h).squeeze().reshape(shape=(-1,))  # you may need to use .concat() for some torch version




