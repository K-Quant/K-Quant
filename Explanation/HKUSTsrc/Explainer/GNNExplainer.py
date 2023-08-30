import gc
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.adam import Adam



device = 'cpu'


class GNNExplainer(nn.Module):
    def __init__(
            self,
            model,
            args
    ):
        super(GNNExplainer, self).__init__()

        self.adj = None
        self.feat = None
        self.mask = None
        self.model = model
        self.args = args
        self.mask_act = args.mask_act
        self.masked_adj = None
        self.optimizer = None

    def set_optimizer(self):
        self.mask.requires_grad = True
        self.optimizer = optim.Adam([self.mask], lr=self.args.lr, weight_decay=0.0)

    def construct_edge_mask(self):
        mask = nn.Parameter(torch.tensor(np.random.rand(self.adj.shape[0],
                                                        self.adj.shape[1],
                                                        self.adj.shape[2]), dtype=torch.float32), requires_grad=False)

        _std = torch.std(mask)
        mask.normal_(1.0, _std)
        # mask = (mask + utils.transpose_3d_tensor(mask)) / 2  # 为了让对称的位置的的mask相同
        mask = mask * self.adj

        _index = torch.t((mask == 0).nonzero())
        mask[_index[0], _index[1], _index[2]] = -100000

        self.mask = mask

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        masked_adj = self.adj * sym_mask
        return masked_adj

    def run_explain(self, feat, adj):
        self.adj = adj
        self.feat = feat
        self.construct_edge_mask()
        self.set_optimizer()

        original_pred = self.model(self.feat, self.adj)
        for epoch in range(self.args.num_epochs):
            self.optimizer.zero_grad()
            self.masked_adj = self._masked_adj()
            masked_pred = self.model(self.feat, self.masked_adj)
            loss = self.loss(original_pred, masked_pred)
            loss.backward(retain_graph=True)
            self.optimizer.step()
        masked_adj = self.masked_adj
        _index = torch.t((masked_adj == 0).nonzero())
        masked_adj[_index[0], _index[1], _index[2]] = -100000
        masked_adj = torch.sigmoid(masked_adj)
        masked_adj = torch.sum(masked_adj, 2) / torch.sum(masked_adj > 0, 2)
        masked_adj = torch.where(
            torch.isnan(masked_adj),
            torch.full_like(masked_adj, 0),
            masked_adj)
        return masked_adj

    def loss(self, original_pred, masked_pred):
        loss_func = torch.nn.L1Loss(reduction='mean')
        pre_loss = loss_func(masked_pred, original_pred)
        return pre_loss

    def mask_density(self):
        mask_sum = torch.sum(torch.sigmoid(self.mask)).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum


if __name__ == '__main__':
    c = np.zeros((3, 3, 2), dtype=np.float)
    c = torch.tensor(c, dtype=torch.float)
    a = torch.tensor([[[0, 1, 1.2],
                       [1, 0, 1.3],
                       [3, 4, 0]],
                      [[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]])
    a[1, 2] = 0

    # index = torch.t((a == 0).nonzero())
    # asd = int(torch.count_nonzero(a))
    # a[index[0], index[1]] = -100000
    # G = nx.from_numpy_matrix(a.detach().numpy())

    print(a)

    # std = nn.init.calculate_gain("relu") * math.sqrt(
    #     2.0 / (3 + 3))
    # print(std)
    # print(math.sqrt(
    #     2.0 / (3 + 3)))
# class GNNExplainer(BaseExplainer, ABC):
#     def __init__(self, model_to_explain, epochs=20, lr=0.1, threshold=0.8):
#         super().__init__(model_to_explain)
#         self.epochs = epochs
#         self.lr = lr
#         self.threshold = threshold
#         self.graph = None
#         self.edge_mask = None
#
#     def _set_masks(self):
#         if self.graph is not None:
#             self.edge_mask = torch.nn.Parameter(torch.randn(self.graph.shape[0],
#                                                             self.graph.shape[1])).requires_grad_(True)
#         else:
#             print('NO GRAPH DATA')
#
#     def _clear_masks(self):
#         self.edge_mask = None
#         self.model_to_explain.edge_mask = None
#
#     def prepare(self, args):
#         """Nothing is done to prepare the GNNExplainer, this happens at every index"""
#         return
#
#     def explain(self, feature, graph):
#         self.graph = graph
#         _2d_graph = GNNExplainer.graph_to_2d(graph)
#         self.model_to_explain.eval()
#         self._clear_masks()
#
#         with torch.no_grad():
#             original_pred = self.model_to_explain(feature, self.graph).requires_grad_(True)
#
#         self._set_masks()
#         optimizer = Adam([self.edge_mask], lr=self.lr)
#
#         for e in range(0, self.epochs):
#             optimizer.zero_grad()
#             edge_mask = torch.sigmoid(self.edge_mask).requires_grad_(True)
#             self.model_to_explain.shift_mask(self.graph, edge_mask)
#             masked_pred = self.model_to_explain(feature, self.graph).requires_grad_(True)
#
#             loss = self._loss(masked_pred.unsqueeze(0), original_pred.unsqueeze(0))
#
#             loss.backward()
#             optimizer.step()
#
#         self.edge_mask = torch.sigmoid(self.edge_mask)*_2d_graph
#
#         return self.edge_mask
#
#     def get_expl_graph_edges(self):
#         expl_graph = torch.zeros(self.edge_mask.size(0), self.edge_mask.size(1))
#         for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
#             link = self.edge_mask.T[i]
#             expl_edge = torch.where((link >= self.threshold))[0]
#             expl_graph[i, expl_edge] = 1
#         return expl_graph
#
#     def _loss(self, masked_pred, original_pred):
#         loss_func = torch.nn.L1Loss(reduction='mean')
#         cce_loss = loss_func(masked_pred, original_pred)
#
#         return cce_loss
#
#     @staticmethod
#     def graph_to_2d(graph):
#         _index = torch.t(torch.nonzero(torch.sum(graph, 2)))
#         _2d_graph = torch.zeros(graph.shape[0], graph.shape[1], device='cpu')
#         _2d_graph[_index[0], _index[1]] = 1
#         return _2d_graph


# if __name__ == '__main__':
#     args = parse_args(config.NRSR_dict)
#
#     # 导入数据
#     data_path = r"{}/{}.pkl".format(args.feature_data_path, args.feature_data_year)
#     f = open(data_path, 'rb')
#     feature_data = pickle.load(f)
#     f.close()
#     graph_data = torch.Tensor(np.load(args.graph_data_path)).to(device)
#
#     # 创建dataloader
#     start_index = len(feature_data.groupby(level=0).size())
#     data_loader = DataLoader(feature_data["feature"], feature_data["label"],
#                              feature_data['market_value'], feature_data['stock_index'],
#                              pin_memory=True, start_index=start_index, device=device)
#
#     with torch.no_grad():
#         num_relation = graph_data.shape[2]  # the number of relations
#         model = NRSR(num_relation=num_relation,
#                      d_feat=args.d_feat,
#                      num_layers=args.num_layers)
#
#         model.to(device)
#         model.load_state_dict(torch.load(args.model_dir + '/model.bin', map_location=device))
#     gnn_explainer = GNNExplainer(model)
#     gnn_explainer_evaluation = FidelityEvaluation()
#     loss1 = gnn_explainer_evaluation.evaluate(gnn_explainer, data_loader,  graph_data, model)
#     print(loss1)
#     del gnn_explainer

# at_explainer = AttentionExplainer(model)
# at_explainer_evaluation = FidelityEvaluation()
# loss2 = at_explainer_evaluation.evaluate(at_explainer, data_loader, graph_data,  model)
#
# print(loss2)

# for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
#     feature, label, market_value, stock_index, index = data_loader.get(slc)
#     expl_g1 = gnn_explainer.explain(feature, graph_data[stock_index][:, stock_index])
#     # expl_g2 = at_explainer.explain(feature, graph_data[stock_index][:, stock_index])
#     break
