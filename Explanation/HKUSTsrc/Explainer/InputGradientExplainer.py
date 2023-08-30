

import numpy as np
import torch.nn as nn
import torch




class InputGradientExplainer(nn.Module):
    def __init__(
            self,
            model
    ):
        super(InputGradientExplainer, self).__init__()
        self.model = model
        self.loss_func = nn.MSELoss()

    def run_explain(self, feat, adj):
        self.model.using_effect_explanation()
        ypred = self.model(feat, adj)

        loss = ypred.sum()
        loss.backward()

        relation_stocks = self.model.relation_stocks

        edge_weight_matrix = InputGradientExplainer.cal_edge_weight(relation_stocks, relation_stocks.grad)
        return edge_weight_matrix

    @staticmethod
    def cal_edge_weight(relation_stocks, grad):
        stocks_num = relation_stocks.shape[0]
        edge_weight_matrix = torch.zeros((stocks_num, stocks_num))
        for idx in range(stocks_num):
            matrix_feat = relation_stocks[idx, :, :]
            metrix_grad = grad[idx, :, :]
            scores_vector = torch.diag(torch.matmul(metrix_grad, matrix_feat.T))
            edge_weight_matrix[idx, :] = scores_vector

        edge_weight_matrix = edge_weight_matrix*10**5
        return edge_weight_matrix

# if __name__ == '__main__':
#     device = 'cpu'
#     args = parse_args(config.NRSR_dict)
#     #
#     #     # 导入数据
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
#
#     for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
#         feature, label, market_value, stock_index, index = data_loader.get(slc)
#         b = graph_data[stock_index][:, stock_index]
#         a = effect_explainer_explain(model, feature, graph_data[stock_index][:, stock_index], label)
#         break













