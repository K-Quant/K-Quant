import json

from utils import *
from SJsrc import *
from HKUSTsrc import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')

    #地址
    parser.add_argument("--data_root", type=str, default="D:\ProjectCodes\K-Quant\Data",
                        help="graph_data root path")
    parser.add_argument("--ckpt_root", type=str, default=r".\pretrianedModel\tmp_ckpt",
                        help="ckpt root path")
    parser.add_argument("--result_root", type=str, default=r".\results",
                        help="explanation resluts root path")
    parser.add_argument('--market_value_path', type=str, default=r'../Data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock_index', type=str, default=r'../Data/csi300_stock_index.npy')
    parser.add_argument('--graph_data_path', default='../Data/csi300_multi_stock2stock_all.npy')
    parser.add_argument('--model_dir', type=str, default=r'.\pretrianedModel\csi300_NRSR_3')

    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=2)

    # 选择预测模型
    parser.add_argument('--graph_model', default='RSR',
                        choices=['RSR', 'GAT'])

    # 选择解释模型
    parser.add_argument('--explainer', default='inputGradientExplainer',
                        choices=['inputGradientExplainer', 'gradExplainer', 'effectExplainer', 'xpathExplainer'])

    parser.add_argument("--relation_type", type=str, default="stock-stock",
                        choices=["stock-stock", "industry", "full"], help="relation type of graph")

    parser.add_argument("--market", type=str, default="A_share",
                        choices=["A_share"], help="market name")

    parser.add_argument('--start_date', type=str, default='2021-01-03')
    parser.add_argument('--end_date', type=str,  default='2021-01-05')
    parser.add_argument('--top_k', type=float, default=3)
    parser.add_argument('--top_p', type=float, default=0.2)

    # check
    parser.add_argument('--stock_list', type=list, default=[])
    parser.add_argument('--date_list', type=list, default=[])
    # parser.add_argument('--top_k', type=int, default=3)

    args = parser.parse_args()
    return args

def run_explanation(args):
    if args.explainer == 'xpathExplainer':
        def select_explanation(args, exp_dict):
            new_exp_dict = {}
            for date in args.date_list:
                new_exp_dict[date] ={}
                exp = exp_dict[date]
                for stock in args.stock_list:
                    if stock not in exp.keys():
                        print(r'No stock {}!'.format(stock))
                    else:
                        new_exp_dict[date][stock] = []
                        for re_stocks, score in exp[stock].items():
                            re_s = re_stocks
                            new_exp_dict[date][stock] += [(re_s, score)]
            return new_exp_dict

        data_df = load_data_df(args)
        model = get_model(args)
        explainer = xPath(graph_model="heterograph", num_layers=1, device=device)
        explanation, score_dict, scores_dict = model.get_explanation(data_df, explainer, None, step_size=1,
                                                                     top_k=args.top_k)

        # relative_stocks_dict = select_explanation(args, explanation)
        relative_stocks_dict = select_top_k_related_stock(explanation,
                                                          k=args.top_k)

        return relative_stocks_dict, scores_dict

    else:
        data_loader = create_data_loaders(args)
        explanation = Explanation(args, data_loader, explainer_name=args.explainer)
        explanation.explain()
        relative_stocks_dict = explanation.check_all_relative_stock()
        score_dict = explanation.check_all_assessment_score()
        return relative_stocks_dict, score_dict


def select_top_k_related_stock(relative_stocks_dict, k=3):
    new_relative_stocks_dict = {}
    for d, s in relative_stocks_dict.items():
        stock_dict = {}
        for os, rs in s.items():
            if len(rs.keys()) <= k:
                if len(rs.keys()) == 0:
                    stock_dict[os] = {}
                else:
                    _k = len(rs)
                    sorted_rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)[:_k]
                    stock_dict[os] = sorted_rs
            else:
                sorted_rs = sorted(rs.items(), key=lambda x:x[1], reverse=True)[:k]
                stock_dict[os] = sorted_rs
        new_relative_stocks_dict[d] = stock_dict
    return new_relative_stocks_dict


if __name__ == '__main__':
    args = parse_args()
    args.start_date = '2022-06-01'
    args.end_date = '2022-06-02'
    args.explainer = 'inputGradientExplainer'
    args.stock_list = ['SH600018']
    args.date_list = ['2022-06-02']
    s_t = time.time()
    relative_stocks_dict, score_dict = run_explanation(args)
    exp_dict = {'relative_stocks_dict': relative_stocks_dict, 'score_dict': score_dict}
    save_path = r'.\results'
    with open(r'{}/{}.json'.format(save_path, args.explainer), 'w') as f:
        json.dump(exp_dict, f)

