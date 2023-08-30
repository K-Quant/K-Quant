import torch
import torch.nn as nn
import dgl


class HomographModel(nn.Module):
    def __init__(self, base_model, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.g = None

    def get_attention(self, graph):
        raise ValueError("please implement cal_attention() in the specific graph model")

    def forward(self, x, index=None):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_rnn(self, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_graph(self, x, index=None, return_subgraph=False):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_predictor(self, x0, x):
        raise ValueError("please implement forward() in the specific graph model ")

    def set_graph(self, rel_encoding, device):
        if len(rel_encoding.shape) == 3:
            rel_encoding = rel_encoding.sum(axis=-1)  # [N, N]
        #idx = (rel_encoding + np.identity(len(rel_encoding))).nonzero()
        idx = (rel_encoding).nonzero()
        self.g = dgl.graph((idx[0], idx[1])).to(device)

    def predict_on_graph(self, g):
        x = g.ndata['nfeat']
        origin_graph = self.g
        self.g = g
        h = self.forward_graph(x)
        pred = self.forward_predictor(x, h)
        self.g = origin_graph
        return pred

class HeterographModel(nn.Module):
    def __init__(self, base_model, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.g = None
        self.target_type = 's' # for stock
        #self.none_feature = nn.Parameter(torch.FloatTensor(size=(1, hidden_size)), requires_grad=False)
        self.use_residual = False
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat


    def get_attention(self, graph):
        raise ValueError("please implement cal_attention() in the specific graph model")

    def forward(self, x, index=None):
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
        #h = {self.target_type: out[:, -1, :], 'none': self.none_feature}
        h = out[:, -1, :]
        return h

    def forward_graph(self, x, index=None, return_subgraph=False):
        raise ValueError("please implement forward() in the specific graph model ")

    def forward_predictor(self, h0, h):
        if self.use_residual:
            return self.fc_out(torch.cat([h0, h], dim=1)).reshape(
                shape=(-1,))  # you may need to use .concat() for some torch version
        else:
            return self.fc_out(h).squeeze().reshape(
                shape=(-1,))  # you may need to use .concat() for some torch version

    def set_graph(self, rel_encoding, device):
        nr = rel_encoding.shape[-1]
        #edges = {('none',str(nr),'none'):([0],[0])} # 'none' type has a virtual node to fit in dgl heterogeneous graph
        edges = {}
        for i in range(nr):
            idx = rel_encoding[:,:,i].nonzero()
            edges[(self.target_type, str(i), self.target_type)] = (idx[0], idx[1])
        #self.none_feature = self.none_feature.to(device)
        self.g = dgl.heterograph(edges).to(device)

    def predict_on_graph(self, g):
        x= g.ndata['nfeat']
        origin_graph = self.g
        self.g = g
        h = self.forward_graph(x)
        pred = self.forward_predictor(x, h)
        self.g = origin_graph
        return pred