import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


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
        device = torch.device(torch.get_device(x))
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


class TGM(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # static+dynamic
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.t_0 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_0.weight)
        # self.t_1 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_1.weight)
        # self.t_2 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def generate_aggregation_graph(self, static, dynamic, n):
        graph_list = [[torch.zeros(n, n,device=static.device)] for i in range(len(dynamic))]
        for static_index in range(static.shape[2]):
            static_graph = static[:, :, static_index]
            # here static_1d have shape [N,N]
            best_bin = -1
            best_fit = -1e5
            for dy_index in range(len(dynamic)):
                dy_graph = dynamic[dy_index]
                fit_score = torch.mean(self.cossim(dy_graph, static_graph))
                if fit_score > best_fit:
                    best_fit = fit_score
                    best_bin = dy_index
            graph_list[best_bin].append(static_graph)
        graph_stack = [torch.stack(x) for x in graph_list]
        return [torch.sum(x, dim=0) for x in graph_stack]

    def forward(self, x, relation_matrix):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # relation_matrix shape 【N,N,type】
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        N = len(x)

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]
        combine_graph = self.generate_aggregation_graph(relation_matrix, [g0, g1, g2], N)
        g0 = g0 + combine_graph[0]
        g1 = g1 + combine_graph[1]
        g2 = g2 + combine_graph[2]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model(nn.Module):
    # first version is the RSR style. including corresponding graph vector in learning
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

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

    def forward(self, x, factor):
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

        N = len(factor)
        factor_hidden = factor.reshape(N, -1, self.factor_num)
        # shape [N, length of single factor vector, factor_num]
        f0 = factor_hidden[:, :, 0]  # shape [N, length of single factor vector]
        f1 = factor_hidden[:, :, 1]  # shape [N, length of single factor vector]
        f2 = factor_hidden[:, :, 2]  # shape [N, length of single factor vector]

        eye = torch.eye(N, N, device=factor_hidden.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        hidden = (hidden_0 + hidden_1 + hidden_2) / 3

        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F(nn.Module):
    # first version is the RSR style. including corresponding graph vector in learning

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

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
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
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
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(x_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(x_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(x_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        hidden = (hidden_0 + hidden_1 + hidden_2) / 3

        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_1(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # sum up three hidden vectors as model v1.1
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        hidden = (hidden_0 + hidden_1 + hidden_2) / 3

        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_2(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # use residual module replace sum up as mdoel v1.2
    # in every part, the attention is computed between ei_x and batch_x
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)
        self.x_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.x_0.weight)
        self.x_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.x_1.weight)
        self.x_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.x_2.weight)
        self.y_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.y_0.weight)
        self.y_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.y_1.weight)
        self.y_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.y_2.weight)
        self.x_r = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.x_r.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size
        x_0 = self.leaky_relu(self.x_0(hidden_0))
        y_0 = self.leaky_relu(self.y_0(hidden_0))

        x_hidden_0 = x_hidden - x_0
        ei_0 = x_hidden_0.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0 = x_hidden_0.unsqueeze(0).repeat(N, 1, 1)
        matrix_0 = torch.cat((ei_0, hidden_batch_0), 2)
        weight_1 = (torch.matmul(matrix_0, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden_0)  # shape N, hidden_size
        x_1 = self.leaky_relu(self.x_1(hidden_1))
        y_1 = self.leaky_relu(self.y_1(hidden_1))

        x_hidden_1 = x_hidden_0 - x_1
        ei_1 = x_hidden_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_1 = x_hidden_1.unsqueeze(0).repeat(N, 1, 1)
        matrix_1 = torch.cat((ei_1, hidden_batch_1), 2)
        weight_2 = (torch.matmul(matrix_1, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden_1)  # shape N, hidden_size
        x_2 = self.leaky_relu(self.x_2(hidden_2))
        y_2 = self.leaky_relu(self.y_2(hidden_2))

        y_r = self.leaky_relu(self.x_r(hidden_2 - x_2))

        hidden = (y_0 + y_1 + y_2 + y_r) / 4

        # hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_3(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # v1.3 follow v1.1 and change RSR style into GATs style, GATs use same parameter to encode both ei and hidden
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0,
                 base_model="GRU", factor_num=3, sparsity=0.10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.W_0 = torch.nn.Parameter(torch.randn(hidden_size, 1))
        self.W_0.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0)
        self.b_0 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0.requires_grad = True
        self.W_1 = torch.nn.Parameter(torch.randn(hidden_size, 1))
        self.W_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1)
        self.b_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1.requires_grad = True
        self.W_2 = torch.nn.Parameter(torch.randn(hidden_size, 1))
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
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, torch.cat((self.W_0, self.W_0), 0)) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, torch.cat((self.W_1, self.W_1), 0)) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, torch.cat((self.W_2, self.W_2), 0)) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        hidden = (hidden_0 + hidden_1 + hidden_2) / 3

        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_4(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vectors together
    # return the cossim as a part of return
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # -----------this part serve as loss, the cossim between different graph-----------
        # -----------we also could use
        loss_1 = self.cossim(g0, g1)
        loss_2 = self.cossim(g0, g2)
        loss_3 = self.cossim(g1, g2)
        loss = float(loss_1.sum() + loss_2.sum() + loss_3.sum())
        # ---------------------------------------------------------------------------------

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_no_F_v1_5(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # v1.5: use 3 different gru to replace
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size * 4, 1)
        # self.t_0 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_0.weight)
        # self.t_1 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_1.weight)
        # self.t_2 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_6(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # 5 heads
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 6, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)
        self.t_3 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_3.weight)
        self.t_4 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_4.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]
        f3 = self.t_3(g_hidden)  # shape [N, hidden_size]
        f4 = self.t_4(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v2(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # v2 series are from v1, and the difference is that v2 model control the sparsity
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        g0 = self.generate_mask(g0, N, self.sparsity)  # shape [N,N]
        g1 = self.generate_mask(g1, N, self.sparsity)  # shape [N,N]
        g2 = self.generate_mask(g2, N, self.sparsity)  # shape [N,N]

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_8(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vectors together
    # return the cossim as a part of return
    # use threshold to control, sparsity set to 0.1
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.1):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        g0 = self.generate_mask(g0, N, self.sparsity)
        g1 = self.generate_mask(g1, N, self.sparsity)
        g2 = self.generate_mask(g2, N, self.sparsity)

        # -----------this part serve as loss, the cossim between different graph-----------
        # -----------we also could use manhattan-------------------------------------------
        loss_1 = self.cossim(g0, g1)
        loss_2 = self.cossim(g0, g2)
        loss_3 = self.cossim(g1, g2)
        loss = float(loss_1.sum() + loss_2.sum() + loss_3.sum())
        # ---------------------------------------------------------------------------------

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_no_F_v1_10(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vectors together
    # return the cossim as a part of return
    # use mean square error as loss return
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden, _ = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        g_hidden = g_hidden[:, -1, :]
        N = len(x)

        # shape [N, length of single factor vector, factor_num]
        f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # -----------this part serve as loss, the cossim between different graph-----------
        # -----------we also could use
        loss_1 = self.cossim(g0, g1)
        loss_2 = self.cossim(g0, g2)
        loss_3 = self.cossim(g1, g2)
        loss = float((loss_1.mean().square() + loss_2.mean().square() + loss_3.mean().square()).sqrt())
        # ---------------------------------------------------------------------------------

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_no_F_v1_11(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vectors together
    # return the cossim as a part of return
    # use three independent rnn to encode temporal attention
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.75):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
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
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fc = nn.Linear(hidden_size * 4, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        N = len(x)

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        # -----------this part serve as loss, the cossim between different graph-----------
        loss_1 = self.cossim(g0, g1) - eye
        loss_2 = self.cossim(g0, g2) - eye
        loss_3 = self.cossim(g1, g2) - eye
        loss = float(loss_1.mean().square() + loss_2.mean().square() + loss_3.mean().square())/3
        # ---------------------------------------------------------------------------------

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_v_3(nn.Module):
    # first version is the RSR style. including corresponding graph vector in learning
    # use factor graph, add linear layer, delete sparsity
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

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
        self.fc = nn.Linear(hidden_size * 4, 1)
        self.t_0 = nn.Linear(27, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_0.weight)
        self.t_1 = nn.Linear(27, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_1.weight)
        self.t_2 = nn.Linear(27, hidden_size)
        torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x, factor):
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

        N = len(factor)
        factor_hidden = factor.reshape(N, -1, self.factor_num)
        # shape [N, length of single factor vector, factor_num]
        f0 = factor_hidden[:, :, 0]  # shape [N, length of single factor vector]
        f1 = factor_hidden[:, :, 1]  # shape [N, length of single factor vector]
        f2 = factor_hidden[:, :, 2]  # shape [N, length of single factor vector]
        f0 = self.t_0(f0)
        f1 = self.t_1(f1)
        f2 = self.t_2(f2)

        eye = torch.eye(N, N, device=factor_hidden.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_13(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # combine v1.5 and v1.6 use 5 seperate RNN to encode graphs
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 6, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_14(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # combine v1.5 and v1.6 use 5 seperate RNN to encode graphs
    # follow v1.13, and use relu after computing cossim matrix
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 6, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.relu(self.sim_matrix(f0, f0) - eye)  # shape [N, N]
        g1 = self.relu(self.sim_matrix(f1, f1) - eye)  # shape [N, N]
        g2 = self.relu(self.sim_matrix(f2, f2) - eye)  # shape [N, N]
        g3 = self.relu(self.sim_matrix(f3, f3) - eye)  # shape [N, N]
        g4 = self.relu(self.sim_matrix(f4, f4) - eye)  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_15(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.13 and delete all mask
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 6, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        valid_weight_0 = g0 * weight_0
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        valid_weight_1 = g1 * weight_1
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        valid_weight_2 = g2 * weight_2
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        valid_weight_3 = g3 * weight_3
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        valid_weight_4 = g4 * weight_4
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_16(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # combine v1.5 and v1.6 use 5 seperate RNN to encode graphs
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 6, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]

        loss = self.cossim(g0, g1).mean().square()
        loss += self.cossim(g0, g2).mean().square()
        loss += self.cossim(g0, g3).mean().square()
        loss += self.cossim(g0, g4).mean().square()
        loss += self.cossim(g1, g2).mean().square()
        loss += self.cossim(g1, g3).mean().square()
        loss += self.cossim(g1, g4).mean().square()
        loss += self.cossim(g2, g3).mean().square()
        loss += self.cossim(g2, g4).mean().square()
        loss += self.cossim(g3, g4).mean().square()

        loss = float((loss / 10).sqrt())
        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_no_F_v1_17(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow the setup of model v1.13 and use double mlp to replace the original fc layer
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.mlp_0 = nn.Linear(hidden_size * 6, hidden_size * 3)
        self.mlp_1 = nn.Linear(hidden_size * 3, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        hidden = self.mlp_0(hidden)
        pred = self.mlp_1(hidden).squeeze()
        return pred


class FC_model_no_F_v1_18(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.13 and add up to 7 heads
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 8, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_19(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.18 and add up to 7 heads LSTM instead
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_1 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_2 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_3 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_4 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_5 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_6 = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm_7 = nn.LSTM(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 8, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.lstm_1(x_hidden_raw)
        g_hidden_1, _ = self.lstm_2(x_hidden_raw)
        g_hidden_2, _ = self.lstm_3(x_hidden_raw)
        g_hidden_3, _ = self.lstm_4(x_hidden_raw)
        g_hidden_4, _ = self.lstm_5(x_hidden_raw)
        g_hidden_5, _ = self.lstm_6(x_hidden_raw)
        g_hidden_6, _ = self.lstm_7(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_20(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.18 and use mlp to generate temporal graph
    # chance to 3 heads for ablation study
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True

        self.fw_0 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_0.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_0)
        self.fb_0 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_0.requires_grad = True
        self.fw_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_1)
        self.fb_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_1.requires_grad = True
        self.fw_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_2)
        self.fb_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_2.requires_grad = True
        self.fw_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_3)
        self.fb_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_3.requires_grad = True
        self.fw_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_4)
        self.fb_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_4.requires_grad = True
        self.fw_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_5)
        self.fb_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_5.requires_grad = True
        self.fw_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.fw_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.fw_6)
        self.fb_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.fb_6.requires_grad = True

        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 8, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]

        N = len(x)

        f0_0 = f0.unsqueeze(1).repeat(1, N, 1)
        f0_1 = f0.unsqueeze(0).repeat(N, 1, 1)
        f0_x = torch.cat((f0_0, f0_1), 2)
        w0 = (torch.matmul(f0_x, self.fw_0) + self.fb_0).squeeze(2)  # shape N,N
        g0 = self.relu(w0)

        f1_0 = f1.unsqueeze(1).repeat(1, N, 1)
        f1_1 = f1.unsqueeze(0).repeat(N, 1, 1)
        f1_x = torch.cat((f1_0, f1_1), 2)
        w1 = (torch.matmul(f1_x, self.fw_1) + self.fb_1).squeeze(2)  # shape N,N
        g1 = self.relu(w1)

        f2_0 = f2.unsqueeze(1).repeat(1, N, 1)
        f2_1 = f2.unsqueeze(0).repeat(N, 1, 1)
        f2_x = torch.cat((f2_0, f2_1), 2)
        w2 = (torch.matmul(f2_x, self.fw_2) + self.fb_2).squeeze(2)  # shape N,N
        g2 = self.relu(w2)

        f3_0 = f3.unsqueeze(1).repeat(1, N, 1)
        f3_1 = f3.unsqueeze(0).repeat(N, 1, 1)
        f3_x = torch.cat((f3_0, f3_1), 2)
        w3 = (torch.matmul(f3_x, self.fw_3) + self.fb_3).squeeze(2)  # shape N,N
        g3 = self.relu(w3)

        f4_0 = f4.unsqueeze(1).repeat(1, N, 1)
        f4_1 = f4.unsqueeze(0).repeat(N, 1, 1)
        f4_x = torch.cat((f4_0, f4_1), 2)
        w4 = (torch.matmul(f4_x, self.fw_4) + self.fb_4).squeeze(2)  # shape N,N
        g4 = self.leaky_relu(w4)

        f5_0 = f5.unsqueeze(1).repeat(1, N, 1)
        f5_1 = f5.unsqueeze(0).repeat(N, 1, 1)
        f5_x = torch.cat((f5_0, f5_1), 2)
        w5 = (torch.matmul(f5_x, self.fw_5) + self.fb_5).squeeze(2)  # shape N,N
        g5 = self.leaky_relu(w5)

        f6_0 = f6.unsqueeze(1).repeat(1, N, 1)
        f6_1 = f6.unsqueeze(0).repeat(N, 1, 1)
        f6_x = torch.cat((f6_0, f6_1), 2)
        w6 = (torch.matmul(f6_x, self.fw_6) + self.fb_6).squeeze(2)  # shape N,N
        g6 = self.leaky_relu(w6)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_18_1(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.13 and add up to 7 heads
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 8, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye

        loss = self.cossim(g0, g1).mean().square()
        loss += self.cossim(g0, g2).mean().square()
        loss += self.cossim(g0, g3).mean().square()
        loss += self.cossim(g0, g4).mean().square()
        loss += self.cossim(g0, g5).mean().square()
        loss += self.cossim(g0, g6).mean().square()
        loss += self.cossim(g1, g2).mean().square()
        loss += self.cossim(g1, g3).mean().square()
        loss += self.cossim(g1, g4).mean().square()
        loss += self.cossim(g1, g5).mean().square()
        loss += self.cossim(g1, g6).mean().square()
        loss += self.cossim(g2, g3).mean().square()
        loss += self.cossim(g2, g4).mean().square()
        loss += self.cossim(g2, g5).mean().square()
        loss += self.cossim(g2, g6).mean().square()
        loss += self.cossim(g3, g4).mean().square()
        loss += self.cossim(g3, g5).mean().square()
        loss += self.cossim(g3, g6).mean().square()
        loss += self.cossim(g4, g5).mean().square()
        loss += self.cossim(g4, g6).mean().square()
        loss += self.cossim(g5, g6).mean().square()

        loss = float((loss / 21).sqrt())

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, loss


class FC_model_no_F_v1_21(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.13 and add up to 9 heads
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_8 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_9 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        self.W_7 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_7.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_7)
        self.b_7 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_7.requires_grad = True
        self.W_8 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_8.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_8)
        self.b_8 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_8.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 10, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)
        g_hidden_7, _ = self.rnn_8(x_hidden_raw)
        g_hidden_8, _ = self.rnn_9(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]
        f7 = g_hidden_7[:, -1, :]
        f8 = g_hidden_8[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye
        g7 = self.sim_matrix(f7, f7) - eye
        g8 = self.sim_matrix(f8, f8) - eye

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        weight_7 = (torch.matmul(matrix, self.W_7) + self.b_7).squeeze(2)  # weight shape N,N
        weight_7 = self.leaky_relu(weight_7)  # relu layer
        index = torch.t((g7 == 0).nonzero())
        valid_weight_7 = g7 * weight_7
        valid_weight_7[index[0], index[1]] = -1000000
        valid_weight_7 = self.softmax1(valid_weight_7)  # shape N,N
        hidden_7 = torch.matmul(valid_weight_7, x_hidden)  # shape N, hidden_size

        weight_8 = (torch.matmul(matrix, self.W_8) + self.b_8).squeeze(2)  # weight shape N,N
        weight_8 = self.leaky_relu(weight_8)  # relu layer
        index = torch.t((g8 == 0).nonzero())
        valid_weight_8 = g8 * weight_8
        valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_8 = self.softmax1(valid_weight_8)  # shape N,N
        hidden_8 = torch.matmul(valid_weight_8, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4,
                            hidden_5, hidden_6, hidden_7, hidden_8), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_22(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # follow v1.13 and add up to 9 heads
    # difference with 21: delete -1000000
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_8 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_9 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        self.W_7 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_7.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_7)
        self.b_7 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_7.requires_grad = True
        self.W_8 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_8.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_8)
        self.b_8 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_8.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 10, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)
        g_hidden_7, _ = self.rnn_8(x_hidden_raw)
        g_hidden_8, _ = self.rnn_9(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]
        f7 = g_hidden_7[:, -1, :]
        f8 = g_hidden_8[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye
        g7 = self.sim_matrix(f7, f7) - eye
        g8 = self.sim_matrix(f8, f8) - eye

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        # valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        # valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        # valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        # valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        # valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        # valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        # valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        weight_7 = (torch.matmul(matrix, self.W_7) + self.b_7).squeeze(2)  # weight shape N,N
        weight_7 = self.leaky_relu(weight_7)  # relu layer
        index = torch.t((g7 == 0).nonzero())
        valid_weight_7 = g7 * weight_7
        # valid_weight_7[index[0], index[1]] = -1000000
        valid_weight_7 = self.softmax1(valid_weight_7)  # shape N,N
        hidden_7 = torch.matmul(valid_weight_7, x_hidden)  # shape N, hidden_size

        weight_8 = (torch.matmul(matrix, self.W_8) + self.b_8).squeeze(2)  # weight shape N,N
        weight_8 = self.leaky_relu(weight_8)  # relu layer
        index = torch.t((g8 == 0).nonzero())
        valid_weight_8 = g8 * weight_8
        # valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_8 = self.softmax1(valid_weight_8)  # shape N,N
        hidden_8 = torch.matmul(valid_weight_8, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4,
                            hidden_5, hidden_6, hidden_7, hidden_8), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v4(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # resGRU and sum all output together, use single GRU encoder.
    def __init__(self, d_feat=6, hidden_size=64, num_layers=3, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.W_0_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_1)
        self.b_0_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_1.requires_grad = True
        self.W_0_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_2)
        self.b_0_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_2.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.x_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.x_0_1_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_1_linear = nn.Linear(hidden_size, hidden_size)
        self.x_0_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_linear = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)

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

    def forward(self, x):
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_0, g_hidden_0 = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0_0 = g_0[:, -1, :]  # [N, 64]
        f0_1 = g_hidden_0[-1, :, :]  # [N, 64]
        f0_2 = g_hidden_0[-2, :, :]  # [N, 64]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0_0.device)
        g0_0 = self.sim_matrix(f0_0, f0_0) - eye  # shape [N, N]
        g0_1 = self.sim_matrix(f0_1, f0_1) - eye
        g0_2 = self.sim_matrix(f0_2, f0_2) - eye

        # 残差第一层用gru最后一层的图，最后一层用gru第一层的图

        ei_0_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_0_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_0_2, hidden_batch_0_2), 2)  # matrix shape N,N,128
        weight_0_2 = self.leaky_relu((torch.matmul(matrix, self.W_0_2) + self.b_0_2).squeeze(2))  # weight shape N,N
        valid_weight_0_2 = g0_0 * weight_0_2
        valid_weight_0_2 = self.softmax1(valid_weight_0_2)  # shape N,N
        hidden_0_2 = torch.matmul(valid_weight_0_2, x_hidden)  # shape N, hidden_size
        output_0_2 = self.leaky_relu(self.y_0_2_linear(hidden_0_2))
        back_0_2 = self.leaky_relu(self.x_0_2_linear(hidden_0_2))  # shape N, hidden_size

        x_hidden_0_1 = x_hidden - back_0_2
        ei_0_1 = x_hidden_0_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0_1 = x_hidden_0_1.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0_1, hidden_batch_0_1), 2)
        weight_0_1 = self.leaky_relu((torch.matmul(matrix, self.W_0_1) + self.b_0_1).squeeze(2))
        valid_weight_0_1 = g0_1 * weight_0_1
        valid_weight_0_1 = self.softmax1(valid_weight_0_1)
        hidden_0_1 = torch.matmul(valid_weight_0_1, x_hidden_0_1)
        output_0_1 = self.leaky_relu(self.y_0_1_linear(hidden_0_1))
        back_0_1 = self.leaky_relu(self.x_0_1_linear(hidden_0_1))  # shape N, hidden_size

        x_hidden_0 = x_hidden_0_1 - back_0_1
        ei_0 = x_hidden_0.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0 = x_hidden_0.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0, hidden_batch_0), 2)
        weight_0 = self.leaky_relu((torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2))
        valid_weight_0 = g0_2 * weight_0
        valid_weight_0 = self.leaky_relu(valid_weight_0)
        hidden_0 = torch.matmul(valid_weight_0, x_hidden_0)
        output_0 = self.leaky_relu(self.y_0_linear(hidden_0))
        back_0 = self.leaky_relu(self.x_0_linear(hidden_0))

        output = output_0_2 + output_0_1 + output_0

        hidden = torch.cat((x_hidden, output), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v4_1(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # resGRU and sum all output together, use single GRU encoder.
    # concat to replace sum up
    def __init__(self, d_feat=6, hidden_size=64, num_layers=3, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
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
        self.W_0_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_1)
        self.b_0_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_1.requires_grad = True
        self.W_0_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_2)
        self.b_0_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_2.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.x_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.x_0_1_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_1_linear = nn.Linear(hidden_size, hidden_size)
        self.x_0_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_linear = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 4, 1)

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

    def forward(self, x):
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_0, g_hidden_0 = self.rnn_1(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0_0 = g_0[:, -1, :]  # [N, 64]
        f0_1 = g_hidden_0[-1, :, :]  # [N, 64]
        f0_2 = g_hidden_0[-2, :, :]  # [N, 64]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0_0.device)
        g0_0 = self.sim_matrix(f0_0, f0_0) - eye  # shape [N, N]
        g0_1 = self.sim_matrix(f0_1, f0_1) - eye
        g0_2 = self.sim_matrix(f0_2, f0_2) - eye

        # 残差第一层用gru最后一层的图，最后一层用gru第一层的图

        ei_0_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_0_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_0_2, hidden_batch_0_2), 2)  # matrix shape N,N,128
        weight_0_2 = self.leaky_relu((torch.matmul(matrix, self.W_0_2) + self.b_0_2).squeeze(2))  # weight shape N,N
        valid_weight_0_2 = g0_0 * weight_0_2
        valid_weight_0_2 = self.softmax1(valid_weight_0_2)  # shape N,N
        hidden_0_2 = torch.matmul(valid_weight_0_2, x_hidden)  # shape N, hidden_size
        output_0_2 = self.leaky_relu(self.y_0_2_linear(hidden_0_2))
        back_0_2 = self.leaky_relu(self.x_0_2_linear(hidden_0_2))  # shape N, hidden_size

        x_hidden_0_1 = x_hidden - back_0_2
        ei_0_1 = x_hidden_0_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0_1 = x_hidden_0_1.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0_1, hidden_batch_0_1), 2)
        weight_0_1 = self.leaky_relu((torch.matmul(matrix, self.W_0_1) + self.b_0_1).squeeze(2))
        valid_weight_0_1 = g0_1 * weight_0_1
        valid_weight_0_1 = self.softmax1(valid_weight_0_1)
        hidden_0_1 = torch.matmul(valid_weight_0_1, x_hidden_0_1)
        output_0_1 = self.leaky_relu(self.y_0_1_linear(hidden_0_1))
        back_0_1 = self.leaky_relu(self.x_0_1_linear(hidden_0_1))  # shape N, hidden_size

        x_hidden_0 = x_hidden_0_1 - back_0_1
        ei_0 = x_hidden_0.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0 = x_hidden_0.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0, hidden_batch_0), 2)
        weight_0 = self.leaky_relu((torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2))
        valid_weight_0 = g0_2 * weight_0
        valid_weight_0 = self.leaky_relu(valid_weight_0)
        hidden_0 = torch.matmul(valid_weight_0, x_hidden_0)
        output_0 = self.leaky_relu(self.y_0_linear(hidden_0))
        back_0 = self.leaky_relu(self.x_0_linear(hidden_0))

        hidden = torch.cat((x_hidden, output_0_2, output_0_1, output_0), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v4_2(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # resGRU and sum all output together, use single GRU encoder.
    # concat to replace sum up
    # 2 head
    def __init__(self, d_feat=6, hidden_size=64, num_layers=3, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.W_0 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0)
        self.b_0 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0.requires_grad = True
        self.W_0_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_1)
        self.b_0_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_1.requires_grad = True
        self.W_0_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_2)
        self.b_0_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_2.requires_grad = True

        self.W_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1)
        self.b_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1.requires_grad = True
        self.W_1_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1_1)
        self.b_1_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1_1.requires_grad = True
        self.W_1_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1_2)
        self.b_1_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1_2.requires_grad = True

        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.x_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_2_linear = nn.Linear(hidden_size, hidden_size)
        self.x_0_1_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_1_linear = nn.Linear(hidden_size, hidden_size)
        # self.x_0_linear = nn.Linear(hidden_size, hidden_size)
        self.y_0_linear = nn.Linear(hidden_size, hidden_size)
        self.x_1_2_linear = nn.Linear(hidden_size, hidden_size)
        self.y_1_2_linear = nn.Linear(hidden_size, hidden_size)
        self.x_1_1_linear = nn.Linear(hidden_size, hidden_size)
        self.y_1_1_linear = nn.Linear(hidden_size, hidden_size)
        # self.x_1_linear = nn.Linear(hidden_size, hidden_size)
        self.y_1_linear = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 7, 1)

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

    def forward(self, x):
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_0, g_hidden_0 = self.rnn_1(x_hidden_raw)
        g_1, g_hidden_1 = self.rnn_2(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0_0 = g_0[:, -1, :]  # [N, 64]
        f0_1 = g_hidden_0[-1, :, :]  # [N, 64]
        f0_2 = g_hidden_0[-2, :, :]  # [N, 64]
        f1_0 = g_1[:, -1, :]
        f1_1 = g_hidden_1[-1, :, :]
        f1_2 = g_hidden_1[-2, :, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0_0.device)
        g0_0 = self.sim_matrix(f0_0, f0_0) - eye  # shape [N, N]
        g0_1 = self.sim_matrix(f0_1, f0_1) - eye
        g0_2 = self.sim_matrix(f0_2, f0_2) - eye
        g1_0 = self.sim_matrix(f1_0, f1_0) - eye  # shape [N, N]
        g1_1 = self.sim_matrix(f1_1, f1_1) - eye
        g1_2 = self.sim_matrix(f1_2, f1_2) - eye

        # 残差第一层用gru最后一层的图，最后一层用gru第一层的图

        ei_0_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_0_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_0_2, hidden_batch_0_2), 2)  # matrix shape N,N,128
        weight_0_2 = self.leaky_relu((torch.matmul(matrix, self.W_0_2) + self.b_0_2).squeeze(2))  # weight shape N,N
        valid_weight_0_2 = g0_0 * weight_0_2
        valid_weight_0_2 = self.softmax1(valid_weight_0_2)  # shape N,N
        hidden_0_2 = torch.matmul(valid_weight_0_2, x_hidden)  # shape N, hidden_size
        output_0_2 = self.leaky_relu(self.y_0_2_linear(hidden_0_2))
        back_0_2 = self.leaky_relu(self.x_0_2_linear(hidden_0_2))  # shape N, hidden_size

        x_hidden_0_1 = x_hidden - back_0_2
        ei_0_1 = x_hidden_0_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0_1 = x_hidden_0_1.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0_1, hidden_batch_0_1), 2)
        weight_0_1 = self.leaky_relu((torch.matmul(matrix, self.W_0_1) + self.b_0_1).squeeze(2))
        valid_weight_0_1 = g0_1 * weight_0_1
        valid_weight_0_1 = self.softmax1(valid_weight_0_1)
        hidden_0_1 = torch.matmul(valid_weight_0_1, x_hidden_0_1)
        output_0_1 = self.leaky_relu(self.y_0_1_linear(hidden_0_1))
        back_0_1 = self.leaky_relu(self.x_0_1_linear(hidden_0_1))  # shape N, hidden_size

        x_hidden_0 = x_hidden_0_1 - back_0_1
        ei_0 = x_hidden_0.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0 = x_hidden_0.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0, hidden_batch_0), 2)
        weight_0 = self.leaky_relu((torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2))
        valid_weight_0 = g0_2 * weight_0
        valid_weight_0 = self.leaky_relu(valid_weight_0)
        hidden_0 = torch.matmul(valid_weight_0, x_hidden_0)
        output_0 = self.leaky_relu(self.y_0_linear(hidden_0))

        # rnn 2
        ei_1_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_1_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_1_2, hidden_batch_1_2), 2)  # matrix shape N,N,128
        weight_1_2 = self.leaky_relu((torch.matmul(matrix, self.W_1_2) + self.b_1_2).squeeze(2))  # weight shape N,N
        valid_weight_1_2 = g1_0 * weight_1_2
        valid_weight_1_2 = self.softmax1(valid_weight_1_2)  # shape N,N
        hidden_1_2 = torch.matmul(valid_weight_1_2, x_hidden)  # shape N, hidden_size
        output_1_2 = self.leaky_relu(self.y_1_2_linear(hidden_1_2))
        back_1_2 = self.leaky_relu(self.x_1_2_linear(hidden_1_2))  # shape N, hidden_size

        x_hidden_1_1 = x_hidden - back_1_2
        ei_1_1 = x_hidden_1_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_1_1 = x_hidden_1_1.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_1_1, hidden_batch_1_1), 2)
        weight_1_1 = self.leaky_relu((torch.matmul(matrix, self.W_1_1) + self.b_1_1).squeeze(2))
        valid_weight_1_1 = g1_1 * weight_1_1
        valid_weight_1_1 = self.softmax1(valid_weight_1_1)
        hidden_1_1 = torch.matmul(valid_weight_1_1, x_hidden_1_1)
        output_1_1 = self.leaky_relu(self.y_1_1_linear(hidden_1_1))
        back_1_1 = self.leaky_relu(self.x_1_1_linear(hidden_1_1))  # shape N, hidden_size

        x_hidden_1 = x_hidden_1_1 - back_1_1
        ei_1 = x_hidden_1.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_1 = x_hidden_1.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_1, hidden_batch_1), 2)
        weight_1 = self.leaky_relu((torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2))
        valid_weight_1 = g1_2 * weight_1
        valid_weight_1 = self.leaky_relu(valid_weight_1)
        hidden_1 = torch.matmul(valid_weight_1, x_hidden_1)
        output_1 = self.leaky_relu(self.y_1_linear(hidden_1))

        hidden = torch.cat((x_hidden, output_0_2, output_0_1, output_0, output_1_1, output_1_2, output_1), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v4_3(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # GRU and sum all output together, use single GRU encoder.
    # concat to replace sum up
    # 2 head
    def __init__(self, d_feat=6, hidden_size=64, num_layers=3, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.W_0 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0)
        self.b_0 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0.requires_grad = True
        self.W_0_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_1)
        self.b_0_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_1.requires_grad = True
        self.W_0_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_0_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_0_2)
        self.b_0_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_0_2.requires_grad = True

        self.W_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1)
        self.b_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1.requires_grad = True
        self.W_1_1 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1_1.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1_1)
        self.b_1_1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1_1.requires_grad = True
        self.W_1_2 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_1_2.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_1_2)
        self.b_1_2 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_1_2.requires_grad = True

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 7, 1)

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

    def forward(self, x):
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_0, g_hidden_0 = self.rnn_1(x_hidden_raw)
        g_1, g_hidden_1 = self.rnn_2(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0_0 = g_0[:, -1, :]  # [N, 64]
        f0_1 = g_hidden_0[-1, :, :]  # [N, 64]
        f0_2 = g_hidden_0[-2, :, :]  # [N, 64]
        f1_0 = g_1[:, -1, :]
        f1_1 = g_hidden_1[-1, :, :]
        f1_2 = g_hidden_1[-2, :, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0_0.device)
        g0_0 = self.sim_matrix(f0_0, f0_0) - eye  # shape [N, N]
        g0_1 = self.sim_matrix(f0_1, f0_1) - eye
        g0_2 = self.sim_matrix(f0_2, f0_2) - eye
        g1_0 = self.sim_matrix(f1_0, f1_0) - eye  # shape [N, N]
        g1_1 = self.sim_matrix(f1_1, f1_1) - eye
        g1_2 = self.sim_matrix(f1_2, f1_2) - eye

        ei_0_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_0_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_0_2, hidden_batch_0_2), 2)  # matrix shape N,N,128
        weight_0_2 = self.leaky_relu((torch.matmul(matrix, self.W_0_2) + self.b_0_2).squeeze(2))  # weight shape N,N
        valid_weight_0_2 = g0_0 * weight_0_2
        valid_weight_0_2 = self.softmax1(valid_weight_0_2)  # shape N,N
        hidden_0_2 = torch.matmul(valid_weight_0_2, x_hidden)  # shape N, hidden_size

        ei_0_1 = x_hidden.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0_1 = x_hidden.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0_1, hidden_batch_0_1), 2)
        weight_0_1 = self.leaky_relu((torch.matmul(matrix, self.W_0_1) + self.b_0_1).squeeze(2))
        valid_weight_0_1 = g0_1 * weight_0_1
        valid_weight_0_1 = self.softmax1(valid_weight_0_1)
        hidden_0_1 = torch.matmul(valid_weight_0_1, x_hidden)

        ei_0 = x_hidden.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_0 = x_hidden.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_0, hidden_batch_0), 2)
        weight_0 = self.leaky_relu((torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2))
        valid_weight_0 = g0_2 * weight_0
        valid_weight_0 = self.leaky_relu(valid_weight_0)
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)

        # rnn 2
        ei_1_2 = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch_1_2 = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei_1_2, hidden_batch_1_2), 2)  # matrix shape N,N,128
        weight_1_2 = self.leaky_relu((torch.matmul(matrix, self.W_1_2) + self.b_1_2).squeeze(2))  # weight shape N,N
        valid_weight_1_2 = g1_0 * weight_1_2
        valid_weight_1_2 = self.softmax1(valid_weight_1_2)  # shape N,N
        hidden_1_2 = torch.matmul(valid_weight_1_2, x_hidden)  # shape N, hidden_size

        ei_1_1 = x_hidden.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_1_1 = x_hidden.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_1_1, hidden_batch_1_1), 2)
        weight_1_1 = self.leaky_relu((torch.matmul(matrix, self.W_1_1) + self.b_1_1).squeeze(2))
        valid_weight_1_1 = g1_1 * weight_1_1
        valid_weight_1_1 = self.softmax1(valid_weight_1_1)
        hidden_1_1 = torch.matmul(valid_weight_1_1, x_hidden)

        ei_1 = x_hidden.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_1 = x_hidden.unsqueeze(0).repeat(N, 1, 1)
        matrix = torch.cat((ei_1, hidden_batch_1), 2)
        weight_1 = self.leaky_relu((torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2))
        valid_weight_1 = g1_2 * weight_1
        valid_weight_1 = self.leaky_relu(valid_weight_1)
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)

        hidden = torch.cat((x_hidden, hidden_0_2, hidden_0_1, hidden_0, hidden_1_1, hidden_1_2, hidden_1), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_23(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # follow v1.13 and add up to 12 heads
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_4 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_5 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_6 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_7 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_8 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_9 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_10 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_11 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_12 = nn.GRU(
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
        self.W_3 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_3.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_3)
        self.b_3 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_3.requires_grad = True
        self.W_4 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_4.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_4)
        self.b_4 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_4.requires_grad = True
        self.W_5 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_5.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_5)
        self.b_5 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_5.requires_grad = True
        self.W_6 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_6.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_6)
        self.b_6 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_6.requires_grad = True
        self.W_7 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_7.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_7)
        self.b_7 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_7.requires_grad = True
        self.W_8 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_8.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_8)
        self.b_8 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_8.requires_grad = True
        self.W_9 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_9.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_9)
        self.b_9 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_9.requires_grad = True
        self.W_10 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_10.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_10)
        self.b_10 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_10.requires_grad = True
        self.W_11 = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
        self.W_11.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W_11)
        self.b_11 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.b_11.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * 13, 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)
        g_hidden_3, _ = self.rnn_4(x_hidden_raw)
        g_hidden_4, _ = self.rnn_5(x_hidden_raw)
        g_hidden_5, _ = self.rnn_6(x_hidden_raw)
        g_hidden_6, _ = self.rnn_7(x_hidden_raw)
        g_hidden_7, _ = self.rnn_8(x_hidden_raw)
        g_hidden_8, _ = self.rnn_9(x_hidden_raw)
        g_hidden_9, _ = self.rnn_10(x_hidden_raw)
        g_hidden_10, _ = self.rnn_11(x_hidden_raw)
        g_hidden_11, _ = self.rnn_12(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]
        f3 = g_hidden_3[:, -1, :]
        f4 = g_hidden_4[:, -1, :]
        f5 = g_hidden_5[:, -1, :]
        f6 = g_hidden_6[:, -1, :]
        f7 = g_hidden_7[:, -1, :]
        f8 = g_hidden_8[:, -1, :]
        f9 = g_hidden_9[:, -1, :]
        f10 = g_hidden_10[:, -1, :]
        f11 = g_hidden_11[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g3 = self.sim_matrix(f3, f3) - eye  # shape [N, N]
        g4 = self.sim_matrix(f4, f4) - eye  # shape [N, N]
        g5 = self.sim_matrix(f5, f5) - eye
        g6 = self.sim_matrix(f6, f6) - eye
        g7 = self.sim_matrix(f7, f7) - eye
        g8 = self.sim_matrix(f8, f8) - eye
        g9 = self.sim_matrix(f9, f9) - eye
        g10 = self.sim_matrix(f10, f10) - eye
        g11 = self.sim_matrix(f11, f11) - eye

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        # valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        # valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        # valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        weight_3 = (torch.matmul(matrix, self.W_3) + self.b_3).squeeze(2)  # weight shape N,N
        weight_3 = self.leaky_relu(weight_3)  # relu layer
        index = torch.t((g3 == 0).nonzero())
        valid_weight_3 = g3 * weight_3
        # valid_weight_3[index[0], index[1]] = -1000000
        valid_weight_3 = self.softmax1(valid_weight_3)  # shape N,N
        hidden_3 = torch.matmul(valid_weight_3, x_hidden)  # shape N, hidden_size

        weight_4 = (torch.matmul(matrix, self.W_4) + self.b_4).squeeze(2)  # weight shape N,N
        weight_4 = self.leaky_relu(weight_4)  # relu layer
        index = torch.t((g4 == 0).nonzero())
        valid_weight_4 = g4 * weight_4
        # valid_weight_4[index[0], index[1]] = -1000000
        valid_weight_4 = self.softmax1(valid_weight_4)  # shape N,N
        hidden_4 = torch.matmul(valid_weight_4, x_hidden)  # shape N, hidden_size

        weight_5 = (torch.matmul(matrix, self.W_5) + self.b_5).squeeze(2)  # weight shape N,N
        weight_5 = self.leaky_relu(weight_5)  # relu layer
        index = torch.t((g5 == 0).nonzero())
        valid_weight_5 = g5 * weight_5
        # valid_weight_5[index[0], index[1]] = -1000000
        valid_weight_5 = self.softmax1(valid_weight_5)  # shape N,N
        hidden_5 = torch.matmul(valid_weight_5, x_hidden)  # shape N, hidden_size

        weight_6 = (torch.matmul(matrix, self.W_6) + self.b_6).squeeze(2)  # weight shape N,N
        weight_6 = self.leaky_relu(weight_6)  # relu layer
        index = torch.t((g6 == 0).nonzero())
        valid_weight_6 = g6 * weight_6
        # valid_weight_6[index[0], index[1]] = -1000000
        valid_weight_6 = self.softmax1(valid_weight_6)  # shape N,N
        hidden_6 = torch.matmul(valid_weight_6, x_hidden)  # shape N, hidden_size

        weight_7 = (torch.matmul(matrix, self.W_7) + self.b_7).squeeze(2)  # weight shape N,N
        weight_7 = self.leaky_relu(weight_7)  # relu layer
        index = torch.t((g7 == 0).nonzero())
        valid_weight_7 = g7 * weight_7
        # valid_weight_7[index[0], index[1]] = -1000000
        valid_weight_7 = self.softmax1(valid_weight_7)  # shape N,N
        hidden_7 = torch.matmul(valid_weight_7, x_hidden)  # shape N, hidden_size

        weight_8 = (torch.matmul(matrix, self.W_8) + self.b_8).squeeze(2)  # weight shape N,N
        weight_8 = self.leaky_relu(weight_8)  # relu layer
        index = torch.t((g8 == 0).nonzero())
        valid_weight_8 = g8 * weight_8
        # valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_8 = self.softmax1(valid_weight_8)  # shape N,N
        hidden_8 = torch.matmul(valid_weight_8, x_hidden)  # shape N, hidden_size

        weight_9 = (torch.matmul(matrix, self.W_9) + self.b_9).squeeze(2)  # weight shape N,N
        weight_9 = self.leaky_relu(weight_9)  # relu layer
        index = torch.t((g9 == 0).nonzero())
        valid_weight_9 = g9 * weight_9
        # valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_9 = self.softmax1(valid_weight_9)  # shape N,N
        hidden_9 = torch.matmul(valid_weight_9, x_hidden)  # shape N, hidden_size

        weight_10 = (torch.matmul(matrix, self.W_10) + self.b_10).squeeze(2)  # weight shape N,N
        weight_10 = self.leaky_relu(weight_10)  # relu layer
        index = torch.t((g10 == 0).nonzero())
        valid_weight_10 = g10 * weight_10
        # valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_10 = self.softmax1(valid_weight_10)  # shape N,N
        hidden_10 = torch.matmul(valid_weight_10, x_hidden)  # shape N, hidden_size

        weight_11 = (torch.matmul(matrix, self.W_11) + self.b_11).squeeze(2)  # weight shape N,N
        weight_11 = self.leaky_relu(weight_11)  # relu layer
        index = torch.t((g11 == 0).nonzero())
        valid_weight_11 = g11 * weight_11
        # valid_weight_8[index[0], index[1]] = -1000000
        valid_weight_11 = self.softmax1(valid_weight_11)  # shape N,N
        hidden_11 = torch.matmul(valid_weight_11, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2, hidden_3, hidden_4,
                            hidden_5, hidden_6, hidden_7, hidden_8, hidden_9, hidden_10, hidden_11), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_24(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_25(nn.Module):
    # use MLP to replace cossim
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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
            names['fw_' + str(i)] = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
            names['fw_' + str(i)].requires_grad = True
            torch.nn.init.xavier_uniform_(names['fw_' + str(i)])
            names['fb_' + str(i)] = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
            names['fb_' + str(i)].requires_grad = True

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def build_att_tensor(self, x, raw, index):
        name = self.__dict__
        gru = name['rnn_' + str(index)].to(x.device)
        g_hidden, _ = gru(raw)
        f = g_hidden[:, -1, :]
        N = len(x)
        f0 = f.unsqueeze(1).repeat(1, N, 1)
        f1 = f.unsqueeze(0).repeat(N, 1, 1)
        fx = torch.cat((f0, f1), 2)
        fw = name['fw_' + str(index)].to(x.device)
        fb = name['fb_' + str(index)].to(x.device)
        w0 = (torch.matmul(fx, fw) + fb).squeeze(2)  # shape N,N
        g = self.leaky_relu(w0)
        ei = x.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        W = name['W_' + str(index)].to(x.device)
        b = name['b_' + str(index)].to(x.device)
        weight = (torch.matmul(matrix, W) + b).squeeze(2)
        weight = self.leaky_relu(weight)  # relu layer
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_26(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # use correlation replace cos sim
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def build_att_tensor(self, x, raw, index):
        name = self.__dict__
        gru = name['rnn_' + str(index)].to(x.device)
        g_hidden, _ = gru(raw)
        f = g_hidden[:, -1, :]
        N = len(x)
        eye = torch.eye(N, N, device=f.device)
        g = torch.corrcoef(f) - eye  # shape [N, N]
        ei = x.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        W = name['W_' + str(index)].to(x.device)
        b = name['b_' + str(index)].to(x.device)
        weight = (torch.matmul(matrix, W) + b).squeeze(2)
        weight = self.leaky_relu(weight)  # relu layer
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_27(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # use one layer MLP to encode hidden vector in graph module
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
        names = self.__dict__
        for i in range(head_num):
            names['encoder_' + str(i)] = nn.Linear(360, hidden_size)
            torch.nn.init.xavier_uniform_(names['encoder_' + str(i)])
            names['W_' + str(i)] = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
            names['W_' + str(i)].require_grad = True
            torch.nn.init.xavier_uniform_(names['W_' + str(i)])
            names['b_' + str(i)] = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
            names['b_' + str(i)].requires_grad = True

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def build_att_tensor(self, x, raw, index):
        name = self.__dict__
        encoder = name['encoder_' + str(index)].to(x.device)
        f = encoder(raw.view(raw.shape[0],-1))
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
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_28(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
        names = self.__dict__
        for i in range(head_num):
            names['rnn_' + str(i)] = nn.LSTM(
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def build_att_tensor(self, x, raw, index):
        name = self.__dict__
        encoder = name['rnn_' + str(index)].to(x.device)
        g_hidden, _ = encoder(raw)
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
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        return hidden

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_29(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # double attention layer
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
        names = self.__dict__
        for i in range(head_num):
            names['rnn_' + str(i)] = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            # 2 level graph encoder
            names['rnn_h' + str(i)] = nn.GRU(
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
            names['W_h' + str(i)] = torch.nn.Parameter(torch.randn((hidden_size * 2), 1))
            names['W_h' + str(i)].require_grad = True
            torch.nn.init.xavier_uniform_(names['W_h' + str(i)])
            names['b_h' + str(i)] = torch.nn.Parameter(torch.tensor(0, dtype=torch.float))
            names['b_h' + str(i)].requires_grad = True

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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
        valid_weight = g * weight
        valid_weight = self.softmax1(valid_weight)  # shape N,N
        hidden = torch.matmul(valid_weight, x)  # shape N, hidden_size
        # second level attention
        gru_h = name['rnn_h' + str(index)].to(x.device)
        g_hidden_h, _ = gru_h(raw)
        f_h = g_hidden_h[:, -1, :]
        g_h = self.sim_matrix(f_h, f_h) - eye
        ei_h = hidden.unsqueeze(1).repeat(1, N, 1)
        hidden_batch_h = hidden.unsqueeze(0).repeat(N, 1, 1)
        matrix_h = torch.cat((ei_h, hidden_batch_h), 2)
        W_h = name['W_h' + str(index)].to(x.device)
        b_h = name['b_h' + str(index)].to(x.device)
        weight_h = (torch.matmul(matrix_h, W_h) + b_h).squeeze(2)
        weight_h = self.leaky_relu(weight_h)
        valid_weight_h = g_h * weight_h
        valid_weight_h = self.softmax1(valid_weight)
        hidden_h = torch.matmul(valid_weight_h, hidden)

        return hidden_h

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_30(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        for h_n in range(self.head_num):
            head_hidden = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_31(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # concat three hidden vector not just sum them up
    # v1.5: use 3 different gru to replace
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.75):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_1 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_2 = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.rnn_3 = nn.GRU(
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
        self.fc = nn.Linear(hidden_size * 4, 1)
        # self.t_0 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_0.weight)
        # self.t_1 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_1.weight)
        # self.t_2 = nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.t_2.weight)

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

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)
        g_hidden_0, _ = self.rnn_1(x_hidden_raw)
        g_hidden_1, _ = self.rnn_2(x_hidden_raw)
        g_hidden_2, _ = self.rnn_3(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        f0 = g_hidden_0[:, -1, :]
        f1 = g_hidden_1[:, -1, :]
        f2 = g_hidden_2[:, -1, :]

        N = len(x)
        # shape [N, length of single factor vector, factor_num]
        # f0 = self.t_0(g_hidden)  # shape [N, hidden_size]
        # f1 = self.t_1(g_hidden)  # shape [N, hidden_size]
        # f2 = self.t_2(g_hidden)  # shape [N, hidden_size]

        eye = torch.eye(N, N, device=f0.device)
        g0 = self.sim_matrix(f0, f0) - eye  # shape [N, N]
        g0 = self.generate_mask(g0, N, self.sparsity)
        g1 = self.sim_matrix(f1, f1) - eye  # shape [N, N]
        g1 = self.generate_mask(g1, N, self.sparsity)
        g2 = self.sim_matrix(f2, f2) - eye  # shape [N, N]
        g2 = self.generate_mask(g2, N, self.sparsity)

        # m0 = self.generate_mask(g0, N, self.sparsity).unsqueeze(2)
        # m1 = self.generate_mask(g1, N, self.sparsity).unsqueeze(2)
        # m2 = self.generate_mask(g2, N, self.sparsity).unsqueeze(2)

        # hm = torch.cat([m0, m1, m2], axis=2)  # shape [N, N, 3]

        ei = x_hidden.unsqueeze(1).repeat(1, N, 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(N, 1, 1)  # shape N,N,64
        matrix = torch.cat((ei, hidden_batch), 2)  # matrix shape N,N,128
        weight_0 = (torch.matmul(matrix, self.W_0) + self.b_0).squeeze(2)  # weight shape N,N
        weight_0 = self.leaky_relu(weight_0)  # relu layer
        index = torch.t((g0 == 0).nonzero())
        valid_weight_0 = g0 * weight_0
        valid_weight_0[index[0], index[1]] = -1000000
        valid_weight_0 = self.softmax1(valid_weight_0)  # shape N,N
        hidden_0 = torch.matmul(valid_weight_0, x_hidden)  # shape N, hidden_size

        weight_1 = (torch.matmul(matrix, self.W_1) + self.b_1).squeeze(2)  # weight shape N,N
        weight_1 = self.leaky_relu(weight_1)  # relu layer
        index = torch.t((g1 == 0).nonzero())
        valid_weight_1 = g1 * weight_1
        valid_weight_1[index[0], index[1]] = -1000000
        valid_weight_1 = self.softmax1(valid_weight_1)  # shape N,N
        hidden_1 = torch.matmul(valid_weight_1, x_hidden)  # shape N, hidden_size

        weight_2 = (torch.matmul(matrix, self.W_2) + self.b_2).squeeze(2)  # weight shape N,N
        weight_2 = self.leaky_relu(weight_2)  # relu layer
        index = torch.t((g2 == 0).nonzero())
        valid_weight_2 = g2 * weight_2
        valid_weight_2[index[0], index[1]] = -1000000
        valid_weight_2 = self.softmax1(valid_weight_2)  # shape N,N
        hidden_2 = torch.matmul(valid_weight_2, x_hidden)  # shape N, hidden_size

        # hidden = (hidden_0+hidden_1+hidden_2)/3

        hidden = torch.cat((x_hidden, hidden_0, hidden_1, hidden_2), 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred


class FC_model_no_F_v1_32(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # use cross graph loss
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=10):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    def cross_loss(self, g_list):
        cross_loss_list = []
        for index_i in range(len(g_list)):
            for index_j in range(index_i+1, len(g_list)):
                cross_loss_list.append(self.cossim(g_list[index_i], g_list[index_j]).mean().square())
        return sum(cross_loss_list)/len(cross_loss_list)

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
        return hidden, g

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        # GAT use homo relation instead
        # device = torch.device(torch.get_device(x))
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        temp_g = []
        for h_n in range(self.head_num):
            head_hidden, g = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)
            temp_g.append(g)

        cross_loss = self.cross_loss(temp_g)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, cross_loss


class FC_model_no_F_v1_33(nn.Module):
    # use different linear layer to encode different graph
    # use independent rnn as uniform graph encoder
    # use correlation in cross graph loss
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="GRU", factor_num=3, sparsity=0.15, head_num=3):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.factor_num = factor_num
        self.sparsity = sparsity
        self.head_num = head_num
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

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size * (1 + head_num), 1)

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

    @staticmethod
    def row_wise_corr(matrix_1,matrix_2):
        return ((matrix_1 * matrix_2).sum(dim=1) / (((matrix_1*matrix_1).sum(dim=1))*((matrix_2 * matrix_2).sum(dim=1))).sqrt()).mean()

    def cross_loss(self, g_list):
        cross_loss_list = []
        for index_i in range(len(g_list)):
            for index_j in range(index_i+1, len(g_list)):
                cross_loss_list.append(self.row_wise_corr(g_list[index_i],g_list[index_j]))
        return sum(cross_loss_list)/len(cross_loss_list)

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
        return hidden, g

    def forward(self, x):
        # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
        # N = the number of stock in current slice
        # F = feature length
        # T = number of days, usually = 60, since F*T should be 360
        # x is the feature of all stocks in one day
        x_hidden_raw = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden_raw = x_hidden_raw.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden_raw)

        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        hidden_vector = [x_hidden]
        temp_g = []
        for h_n in range(self.head_num):
            head_hidden, g = self.build_att_tensor(x_hidden, x_hidden_raw, h_n)
            hidden_vector.append(head_hidden)
            temp_g.append(g)

        cross_loss = self.cross_loss(temp_g)

        hidden = torch.cat(hidden_vector, 1)
        # now hidden shape (N,hidden_size*2) stores all new embeddings
        pred = self.fc(hidden).squeeze()
        return pred, cross_loss