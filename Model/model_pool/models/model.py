import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import sys
sys.path.append("..")
from utils.utils import cal_cos_similarity


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=1, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d' % i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d' % i, nn.Linear(
                360 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d' % i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()


class HIST(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU", K=3):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)

        self.fc_out_ps = nn.Linear(hidden_size, 1)
        self.fc_out_hs = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.K = K

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def forward(self, x, concept_matrix, market_value):
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # device = torch.device(torch.get_device(x))
        device = x.device
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]
        # get the last layer embeddings

        # Predefined Concept Module

        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, concept_matrix.shape[1])
        # make the market value matrix the same size as the concept matrix by repeat
        # market value matrix shape: (N, number of pre define concepts)
        stock_to_concept = concept_matrix * market_value_matrix
        # torch.sum generate (1, number of pre define concepts) -> repeat (N, number of predefine concepts)
        # 对应每个concept 得到其相关所有股票市值的和, sum在哪个维度上操作，哪个维度被压缩成1
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        # mul得到结果 （N，number of predefine concepts），每个股票对应的概念不再是0或1，而是0或其相关股票市值之和
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        # 所有位置+1，防止除法报错
        stock_to_concept_sum = stock_to_concept_sum + (
            torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device))
        # 做除法，得到每个股票对应的在concepts上的权重，对应公式4
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        # stock_to_concept transpose (number of predefine concept, N) x_hidden(N, the output of gru)
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        # hidden here is the embeddings of all predefine concept (number of concept, the output of gru)
        # 至此concept的embeddings初始化完成，对应论文中公式5
        hidden = hidden[hidden.sum(1) != 0]
        # stock_to_concept (N, number of concept) 对应embeddings相乘相加
        stock_to_concept = x_hidden.mm(torch.t(hidden))
        # stock_to_concept = cal_cos_similarity(x_hidden, hidden)
        # 对dim0作softmax， stock_to_concept (N, number of concept)，得到不同股票在同一concept上的权重
        stock_to_concept = self.softmax_s2t(stock_to_concept)
        # hidden shape (number of concept, output of gru) now hidden have the embedding of all concepts
        # 使用新得到的权重更新hidden中concept的embeddings
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        # 计算x_hidden和hidden的cos sim concept_to_stock shape (N, number of concept)
        concept_to_stock = cal_cos_similarity(x_hidden, hidden)
        # softmax on dim1, (N, number of concept) 得到同一股票在不同concept上的权重，公式6
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        # p_shared_info (N, output of gru) 公式7的累加部分
        # 过三个不同的linear层输出三个不同的tensor
        # output_ps 通过leaky_relu，公式7
        p_shared_info = concept_to_stock.mm(hidden)
        p_shared_info = self.fc_ps(p_shared_info)

        p_shared_back = self.fc_ps_back(p_shared_info)
        output_ps = self.fc_ps_fore(p_shared_info)
        output_ps = self.leaky_relu(output_ps)

        pred_ps = self.fc_out_ps(output_ps).squeeze()

        # Hidden Concept Module
        h_shared_info = x_hidden - p_shared_back
        hidden = h_shared_info
        # compute the cos sim between stocks and h_con(h_con generated from stocks, so cos sim with itself)
        h_stock_to_concept = cal_cos_similarity(h_shared_info, hidden)

        dim = h_stock_to_concept.shape[0]
        diag = h_stock_to_concept.diagonal(0)
        # delete itself
        h_stock_to_concept = h_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        # row = torch.linspace(0,dim-1,dim).to(device).long()
        # column = h_stock_to_concept.argmax(1)
        # split dim-1 into dim pieces, then reshape to (dim, 1) -> repeat (dim, K) -> reshape (1, dim*K)
        row = torch.linspace(0, dim - 1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        # found column index of topk value, and reshape to (1, dim*K)
        column = torch.topk(h_stock_to_concept, self.K, dim=1)[1].reshape(1, -1)
        mask = torch.zeros([h_stock_to_concept.shape[0], h_stock_to_concept.shape[1]], device=h_stock_to_concept.device)
        # set the topk position mask to 1
        mask[row, column] = 1
        h_stock_to_concept = h_stock_to_concept * mask
        # add the original embedding h_stock_to_concept (N,N)
        h_stock_to_concept = h_stock_to_concept + torch.diag_embed((h_stock_to_concept.sum(0) != 0).float() * diag)
        # hidden shape (the length of embedding, N)*(N,N) -> transpose (N, the length of embedding)
        hidden = torch.t(h_shared_info).mm(h_stock_to_concept).t()
        # delete concepts that have no connections
        hidden = hidden[hidden.sum(1) != 0]

        h_concept_to_stock = cal_cos_similarity(h_shared_info, hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock)
        h_shared_info = h_concept_to_stock.mm(hidden)
        h_shared_info = self.fc_hs(h_shared_info)

        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)
        output_hs = self.leaky_relu(output_hs)
        pred_hs = self.fc_out_hs(output_hs).squeeze()

        # Individual Information Module
        individual_info = x_hidden - p_shared_back - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        pred_indi = self.fc_out_indi(output_indi).squeeze()
        # Stock Trend Prediction
        all_info = output_ps + output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        return pred_all


class GRU(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x shape N, F*T
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.gru(x)
        # deliver the last layer as output
        return self.fc(out[:, -1, :]).squeeze()


class LSTM(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x shape (N, F*T)
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


class GAT(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
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

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transfer = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def self_attention(self, x):
        # compute attention between each stock in the day
        x = self.transfer(x)
        stock_num = x.shape[0]
        hidden_size = x.shape[1]
        e_x = x.expand(stock_num, stock_num, hidden_size)  # shape N*N*h
        e_y = torch.transpose(e_x, 0, 1)  # shape N*N*h
        attention_in = torch.cat((e_x, e_y), 2).view(-1, hidden_size * 2)  # shape N*N*2h -> 2N*2h
        self.a_t = torch.t(self.a)  # shape 1*2h
        attention_out = self.a_t.mm(torch.t(attention_in)).view(stock_num, stock_num)  # shape 1*2N -> N*N
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x shape （N，F*T）
        x = x.reshape(len(x), self.d_feat, -1)  # N, F, T
        x = x.permute(0, 2, 1)  # N, T, F
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # N*h
        att_weight = self.self_attention(hidden)  # N*N
        hidden = att_weight.mm(hidden) + hidden  # N*h + N*h
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


class RSR(nn.Module):
    def __init__(self, num_relation, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.W = torch.nn.Parameter(torch.randn((hidden_size * 2) + num_relation, 1))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, relation_matrix):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # device = torch.device(torch.get_device(x))
        device = x.device
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        # get the last layer embeddings
        # update embedding using relation_matrix
        # relation matrix shape [N, N]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)  # matrix shape N,N,64+关系数
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
        mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        mask[index[0], index[1]] = 1
        # valid_weight = mask*weight
        # valid_weight = self.softmax1(valid_weight)
        temp_weight = mask * weight
        index_2 = torch.t((temp_weight == 0).nonzero())
        temp_weight[index_2[0], index_2[1]] = -10000
        valid_weight = self.softmax1(temp_weight)  # N,N
        valid_weight = valid_weight * mask
        hidden = torch.matmul(valid_weight, x_hidden)
        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        pred_all = self.fc(hidden).squeeze()
        return pred_all


class SFM(nn.Module):
    def __init__(self, d_feat=6, output_dim=1, freq_dim=10, hidden_size=64, dropout_W=0.0, dropout_U=0.0,
                 device="cpu", ):
        super().__init__()

        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc = nn.Linear(self.output_dim, 1)

        self.states = []

    def forward(self, input):
        input = input.reshape(len(input), self.input_dim, -1)  # [N, F, T]
        input = input.permute(0, 2, 1)  # [N, T, F]
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]  # noqa: F841
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time, None, None,
                       None, ]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants


class ALSTM(nn.Module):
    # need comments
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            rnn = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = rnn(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layer,
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

    def forward(self, x):
        x = x.view(len(x), self.input_size, -1)  # shape N*d_feat*days
        x = x.permute(0, 2, 1)  # shape N*days*d_feat
        rnn_out, _ = self.rnn(self.net(x))  # shape N*days*hidden_size
        attention_score = self.att_net(rnn_out)  # shape N*days*1
        out_att = torch.mul(rnn_out, attention_score)  # shape N*days*hidden_size
        out_att = torch.sum(out_att, dim=1)  # shape N*hidden_size
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # N, 2*hidden_size -> N, 1
        return out[..., 0]


class relation_GATs(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.W = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, relation_matrix):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        # get the last layer embeddings
        # update embedding using relation_matrix
        # relation matrix shape [N, N]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,64*2
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        mask = torch.sum(relation_matrix, 2)  # mask that could have relation value
        index = torch.t((mask == 0).nonzero())
        valid_weight = mask * weight
        valid_weight[index[0], index[1]] = -1000000
        valid_weight = self.softmax1(valid_weight)

        # valid_weight = self.softmax1(temp_weight)  # N,N
        hidden = torch.matmul(valid_weight, x_hidden)  # N, N, hidden_size
        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class relation_GATs_3heads(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.W_0 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0)
        self.b_0 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0.requires_grad = True

        self.W_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1)
        self.b_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1.requires_grad = True

        self.W_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_2)
        self.b_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_2.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, relation_matrix):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        # get the last layer embeddings
        # update embedding using relation_matrix
        # relation matrix shape [N, N]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,64+关系数
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        # index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))  # store all index that have relation
        # mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        # mask[index[0], index[1]] = 1
        mask = torch.sum(relation_matrix, 2)  # mask that could have relation value
        index = torch.t((mask == 0).nonzero())
        valid_weight_0 = mask * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)
        # temp_weight = mask*weight
        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        valid_weight_1 = mask * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        valid_weight_2 = mask * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)

        # valid_weight = self.softmax1(temp_weight)  # N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # N, N, hidden_size
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # N, N, hidden_size
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # N, N, hidden_size
        hidden = (hidden_0 + hidden_1 + hidden_2) / 3
        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class KEnhance(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # add RSR and ALSTM as a part of hidden vector
    def __init__(self, num_relation, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=7):
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

        self.W = torch.nn.Parameter(torch.randn((hidden_size * 2) + num_relation, 1))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b.requires_grad = True

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (3 + head_num), 1)
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

    def forward(self, x, relation_matrix):
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(self.net(x_hidden_raw))

        # ------ALSTM module--------------
        att_score = self.att_net(x_hidden)
        alstm_out = torch.mul(x_hidden, att_score)
        alstm_out = torch.sum(alstm_out, dim=1)  # shape N*hidden_size

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden, alstm_out]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        # ---------RSR-----------
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

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred
