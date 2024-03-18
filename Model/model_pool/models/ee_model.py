import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.insert(0, str(DIRNAME.parent.parent))

class event_embedding_model(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # add RSR and ALSTM as a part of hidden vector
    def __init__(self, num_relation, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=1):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
        self.base_model = base_model
        self.num_layers = num_layers
        self.dropout = dropout
        names = self.__dict__
        for i in range(head_num):
            names['rnn_' + str(i)] = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            names['W_' + str(i)] = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
            names['W_' + str(i)].require_grad = True
            torch.nn.init.xavier_uniform_(names['W_' + str(i)])
            names['b_' + str(i)] = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
            names['b_' + str(i)].requires_grad = True

        # self.rnn = nn.GRU(
        #     input_size=d_feat,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout,
        # )

        self.ee_rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.W = torch.nn.Parameter(torch.randn((hidden_size * 2) + num_relation, 1))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b.requires_grad = True

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (4 + head_num), 1)
        # here the hidden_size 1 from text embedding, one from x_hidden, one from relation, one from attentive
        self._build_model()

    @staticmethod
    def sim_matrix(a, b, eps=1e-6):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @staticmethod
    def generate_mask(a, N, sparsity=0.1):
        v, i = torch.topk(a.flatten(), int(N * N * sparsity))
        return torch.ones(N, N, device=a.device) * (a >= v[-1])

    def _build_model(self):
        try:
            rnn = getattr(nn, self.base_model.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.base_model) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.d_feat, out_features=self.hidden_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = rnn(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hidden_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def build_att_tensor(self, x, raw, index):
        name = self.__dict__
        gru = name['rnn_' + str(index)].to(x.device)
        g_hidden, _ = gru(raw)
        f = g_hidden[:, -1, :]
        N = len(x)
        eye = torch.eye(N, N, device=f.device)
        g = self.sim_matrix(f, f) - eye  # shape [N, N]
        ei = x.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        W = name['W_' + str(index)].to(x.device)
        b = name['b_' + str(index)].to(x.device)
        weight = (torch.matmul(matrix, W) + b).squeeze(2)
        weight = self.leaky_relu(weight)  # relu layer
        # valid_weight = g * weight
        index = torch.t((g == 0).nonzero())
        valid_weight = g * weight
        valid_weight[index[0], index[1]] = -1e10
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x, relation_matrix, ee):
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(self.net(x_hidden_raw))

        ee_hidden, _ = self.ee_rnn(ee.unsqueeze(-1))
        ee_hidden = ee_hidden[:, -1, :]

        # ------ALSTM module--------------
        att_score = self.att_net(x_hidden)
        alstm_out = torch.mul(x_hidden, att_score)
        alstm_out = torch.sum(alstm_out, dim=1)  # shape N*hidden_size

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden, alstm_out]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        # ---------Temporal Relation Module-----------
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)  # matrix shape N,N,64+关系数
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
        mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        mask[index[0], index[1]] = 1
        temp_weight = mask * weight
        index_2 = torch.t((temp_weight == 0).nonzero())
        temp_weight[index_2[0], index_2[1]] = -100000
        valid_weight = self.softmax1(temp_weight)  # N,N
        valid_weight = valid_weight * mask
        relation_hidden = torch.matmul(valid_weight, x_hidden)
        hidden_vector.append(relation_hidden)

        hidden_vector.append(ee_hidden)  # concat the event embeddings

        hidden = torch.cat(hidden_vector, 1)  # shape N, 640
        # hidden = torch.cat([hidden, ee], 1)  # shape N, 640+64

        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred
