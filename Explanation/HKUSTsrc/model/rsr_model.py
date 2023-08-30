import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class RSR(nn.Module):
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
        self.W = torch.nn.Parameter(torch.randn((hidden_size*2)+1, 1, dtype = torch.double))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)
        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.double))
        self.b.requires_grad = True
        # Fan in and fan out can not be computed for tensor with fewer than 2 dimensions
        # torch.nn.init.xavier_uniform_(self.b)
        self.leaky_relu = nn.LeakyReLU()
        # if use for loop in forward, change dim to 0
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size*2, 1)


    def forward(self, x, relation_matrix):
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
        matrix = torch.cat((ei, hidden_batch, relation_matrix.unsqueeze(2)), 2)
        matrix.double()# matrix shape N,N,129
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        index = torch.t(torch.nonzero(relation_matrix))
        mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
        mask[index[0], index[1]] = 1
        temp_weight = mask*weight
        index_2 = torch.t((temp_weight == 0).nonzero())
        temp_weight[index_2[0], index_2[1]] = -float('inf')
        valid_weight = self.softmax1(temp_weight)  # N,N
        hidden = torch.matmul(valid_weight, x_hidden)
        hidden = torch.cat((x_hidden, hidden), 1)
        # now hidden shape (N,64) stores all new embeddings
        pred_all = self.fc(hidden).squeeze()
        return pred_all


class NRSR(nn.Module):
    def __init__(self, num_relation, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        self.relation_stocks = None
        self.attention_weight = None
        self.using_effect = False
        self.using_attention = False
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.W = torch.nn.Parameter(torch.randn((hidden_size*2)+num_relation, 1, dtype=torch.float32))
        self.W.requires_grad = True
        torch.nn.init.xavier_uniform_(self.W)

        self.b = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.b.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x, relation_matrix):
        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]  # [N, 64]
        ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
        hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1) # shape N,N,64
        # 分离叶子节点
        hidden_batch = hidden_batch.detach()
        hidden_batch.requires_grad = True

        matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)# matrix shape N,N,64+关系数
        weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
        weight = self.leaky_relu(weight)  # relu layer
        mask = torch.sum(relation_matrix, 2)# mask that could have relation value
        index_2 = torch.t((mask == 0).nonzero())
        valid_weight = mask * weight

        if self.using_effect:
            self.relation_stocks = hidden_batch
        if self.using_attention:
            self.attention_weight = valid_weight

        valid_weight[index_2[0], index_2[1]] = -1000000
        valid_weight = self.softmax1(valid_weight)
        hidden = torch.matmul(valid_weight, x_hidden)
        hidden = torch.cat((x_hidden, hidden), 1)
        pred_all = self.fc(hidden).squeeze()

        if self.using_effect:
            self.relation_stocks = hidden_batch
        if self.using_attention:
            self.attention_weight = valid_weight
        return pred_all

    def using_effect_explanation(self):
        self.using_effect = True

    def using_attention_explanation(self):
        self.using_attention = True

    def reset(self):
        self.using_effect = False
        self.using_attention = False
        self.relation_stocks = None
        self.attention_weight = None




    # def forward(self, x, relation_matrix):
    #     x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    #     x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
    #     x_hidden, _ = self.rnn(x_hidden)
    #     x_hidden = x_hidden[:, -1, :] # [N, 64]
    #
    #     ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
    #     hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
    #     matrix = torch.cat((ei, hidden_batch, relation_matrix), 2) # matrix shape N,N,64+关系数
    #
    #     weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
    #     weight = self.leaky_relu(weight)
    #     index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
    #     mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device='cpu')
    #     mask[index[0], index[1]] = 1
    #     # relu layer
    #     # valid_weight = mask*weight
    #     # valid_weight = self.softmax1(valid_weight)
    #     temp_weight = mask*weight
    #     index_2 = torch.t((temp_weight == 0).nonzero())
    #     temp_weight[index_2[0], index_2[1]] = -10000
    #
    #     self.valid_weight = self.softmax1(temp_weight)
    #     valid_weight = self.softmax1(temp_weight)
    #     if self.edge_mask is not None:
    #         valid_weight = valid_weight * self.edge_mask
    #
    #     hidden = torch.matmul(self.valid_weight, x_hidden)
    #     hidden = torch.cat((x_hidden, hidden), 1)
    #     # now hidden shape (N,64) stores all new embeddings
    #     pred_all = self.fc(hidden).squeeze()
    #     return pred_all











    # def forward(self, x, relation_matrix):
    #     # 注意，使用multi relation不能加负无穷大来使softmax后的值为0，不然会发生梯度回传为nan
    #     # 此版forward用来测试非相关股票矩阵点赋-10000的情况（默认为0）
    #     # N = the number of stock in current slice
    #     # F = feature length
    #     # T = number of days, usually = 60, since F*T should be 360
    #     # x is the feature of all stocks in one day
    #     device = torch.device(torch.get_device(x))
    #     x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
    #     x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
    #     x_hidden, _ = self.rnn(x_hidden)
    #     x_hidden = x_hidden[:, -1, :]  # [N, 64]
    #     # get the last layer embeddings
    #     # update embedding using relation_matrix
    #     # relation matrix shape [N, N]
    #     ei = x_hidden.unsqueeze(1).repeat(1, relation_matrix.shape[0], 1)  # shape N,N,64
    #     hidden_batch = x_hidden.unsqueeze(0).repeat(relation_matrix.shape[0], 1, 1)  # shape N,N,64
    #     matrix = torch.cat((ei, hidden_batch, relation_matrix), 2)  # matrix shape N,N,64+关系数
    #     weight = (torch.matmul(matrix, self.W) + self.b).squeeze(2)  # weight shape N,N
    #     weight = self.leaky_relu(weight)  # relu layer
    #     index = torch.t(torch.nonzero(torch.sum(relation_matrix, 2)))
    #     mask = torch.zeros(relation_matrix.shape[0], relation_matrix.shape[1], device=x_hidden.device)
    #     mask[index[0], index[1]] = 1
    #     # valid_weight = mask*weight
    #     # valid_weight = self.softmax1(valid_weight)
    #     temp_weight = mask*weight
    #     index_2 = torch.t((temp_weight == 0).nonzero())
    #     temp_weight[index_2[0], index_2[1]] = -10000
    #     valid_weight = self.softmax1(temp_weight)  # N,N
    #     valid_weight = valid_weight*mask
    #     hidden = torch.matmul(valid_weight, x_hidden)
    #     hidden = torch.cat((x_hidden, hidden), 1)
    #     # now hidden shape (N,64) stores all new embeddings
    #     pred_all = self.fc(hidden).squeeze()
    #     return pred_all
    #
