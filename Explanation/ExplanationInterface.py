import json

from Explanation.utils import *
from Explanation.SJsrc import *
from Explanation.HKUSTsrc import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cpu')

    # 地址
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
    parser.add_argument('--relation_name_list_file', type=str, default=r'..\Data\relation_name_list.json')



    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=2)

    # 选择预测模型
    parser.add_argument('--graph_model', default='NRSR')

    # 选择解释模型
    parser.add_argument('--explainer', default='inputGradientExplainer',
                        choices=['inputGradientExplainer', 'gradExplainer', 'effectExplainer', 'xpathExplainer'])

    parser.add_argument("--relation_type", type=str, default="stock-stock",
                        choices=["stock-stock", "industry", "full"], help="relation type of graph")

    parser.add_argument("--market", type=str, default="A_share",
                        choices=["A_share"], help="market name")

    parser.add_argument('--start_date', type=str, default='2021-01-03')
    parser.add_argument('--end_date', type=str, default='2021-01-05')
    parser.add_argument('--top_k', type=float, default=3)
    parser.add_argument('--top_p', type=float, default=0.2)

    # check
    parser.add_argument('--stock_list', type=list, default=[])
    parser.add_argument('--date_list', type=list, default=[])
    # parser.add_argument('--top_k', type=int, default=3)
    args = parser.parse_known_args()[0]
    return args


def run_explanation(args):
    if args.explainer == 'xpathExplainer':
        def select_explanation(args, exp_dict):
            new_exp_dict = {}
            for date in args.date_list:
                new_exp_dict[date] = {}
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


def run_input_gradient_explanation(args):
    if not args.explainer == 'inputGradientExplainer':
        return None
    data_loader = create_data_loaders(args)
    explanation = Explanation(args, data_loader, explainer_name=args.explainer)
    exp_result_dict = explanation.explain()
    exp_result_dict = check_all_relative_stock(args, exp_result_dict)

    return exp_result_dict, explanation


def search_stocks_prediction_explanation(args, exp_result_dict):
    result_subset = {}
    for stock_id in args.stock_list:
        result_subset[stock_id] = {}
        for date, result_data in exp_result_dict.items():
            # Check if date is within the specified range
            if args.start_date <= date <= args.end_date:
                if stock_id in result_data.keys():
                    result_subset[stock_id][date] = result_data[stock_id]
    return result_subset


def run_xpath_explanation(args, get_fidelity=False, top_k=3):
    data_loader = create_data_loaders(args)
    explanation = Explanation(args, data_loader, explainer_name=args.explainer)
    with open(args.relation_name_list_file, 'r') as json_file:
        _relation_name_list = json.load(json_file)
    res = explanation.explain_xpath(stock_list=args.stock_list, get_fidelity=get_fidelity,
                                         top_k=top_k, relation_list=_relation_name_list)
    return res


def check_all_relative_stock(args, exp_result_dict):
    _stock_index = np.load(args.stock_index, allow_pickle=True).item()
    with open(args.relation_name_list_file, 'r') as json_file:
        _relation_name_list = json.load(json_file)
    index_to_stock_id = {index: stock_id for stock_id, index in _stock_index.items()}

    relative_stocks_dict = {}
    for date in exp_result_dict.keys():
        exp_graph = exp_result_dict[date]['expl_graph']
        stock_index_in_adj = exp_result_dict[date]['stock_index_in_adj']
        relative_stocks_dict[date] = {}

        num_stocks, _, num_relations = exp_graph.shape

        for i in range(num_stocks):
            stock_id = index_to_stock_id[stock_index_in_adj[i]]  # 获取股票编号
            relative_stocks_dict[date][stock_id] = {}
            related_stocks = []

            for j in range(num_stocks):
                other_stock_id = index_to_stock_id[stock_index_in_adj[j]]
                scores = exp_graph[i, j, :]
                total_score = scores.sum()
                relative_scores = {rel_name: score for rel_name, score in zip(_relation_name_list, scores) if score != 0}

                if relative_scores:
                    related_stocks.append({
                        'relative_stock_id': other_stock_id,
                        'total_score': total_score,
                        'individual_scores': relative_scores
                    })

            # Sort related_stocks based on total_score
            related_stocks.sort(key=lambda x: x['total_score'], reverse=True)
            # Keep only the top three related stocks
            top_three_related_stocks = related_stocks[:3]

            for entry in top_three_related_stocks:
                other_stock_id = entry['relative_stock_id']
                relative_stocks_dict[date][stock_id][other_stock_id] = {
                    'total_score': entry['total_score'],
                    'individual_scores': entry['individual_scores']
                }

    return relative_stocks_dict


def evaluate_fidelity(explanation, exp_result_dict, p=0.2):
    if exp_result_dict is None:
        print("exp_result_dict is None")
    evaluation_results = {}
    for date, exp in exp_result_dict.items():
        expl_graph = exp['expl_graph']
        origin_graph = exp['origin_graph']
        feature = exp['feature']
        fidelity = explanation.evaluate_explanation(feature, expl_graph, origin_graph, p=p)
        evaluation_results[date] = fidelity
    return evaluation_results


def get_assessment(args):
    data_loader = create_data_loaders(args)
    explanation = Explanation(args, data_loader, explainer_name=args.explainer)
    reliability, stability = explanation.cal_reliability_stability()
    return reliability, stability


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
                sorted_rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)[:k]
                stock_dict[os] = sorted_rs
        new_relative_stocks_dict[d] = stock_dict
    return new_relative_stocks_dict


def get_results(args, start_date, end_date, explainer, check_stock_list, check_date_list):
    args.start_date = start_date
    args.end_date = end_date
    args.explainer = explainer
    args.stock_list = check_stock_list
    args.date_list = check_date_list
    relative_stocks_dict, score_dict = run_explanation(args)

    for date, stocks_score in score_dict.items():
        if date not in check_date_list:
            continue
        for stock, score in stocks_score.items():
            if stock not in check_stock_list:
                continue
            print(r'--------------------------------------------------------------------')
            print(r'股票 {} 解释结果如下：'.format(stock))
            print(r'最相关的股票与得分{}'.format(relative_stocks_dict[date][stock]))
            print(r'对该解释结果的评价如下：')
            print(r'总得分：{}，保真度得分：{}，准确性得分：{}， 稀疏性得分：{}'.format(
                score['score'],
                score['f_score'],
                score['a_score'],
                score['s_score']

            ))


if __name__ == '__main__':
    args = parse_args()
    args.start_date = '2022-06-01'
    args.end_date = '2022-06-05'
    args.stock_list = ['SH600018']
    # args.date_list = ['2022-06-02']

    # for inputGradient:
    args.explainer = 'inputGradientExplainer'
    exp_result_dict, explanation = run_input_gradient_explanation(args)
    exp_result_dict = search_stocks_prediction_explanation(args, exp_result_dict)
    # fidelity = evaluate_fidelity(explanation, exp_result_dict, 0.2)
    print(exp_result_dict)

    # for xpath:
    # args.explainer = 'xpathExplainer'
    # args.stock_list = ['SH600018']
    # exp_result_dict = run_xpath_explanation(args, get_fidelity=False, top_k=3)
    # print(exp_result_dict)
    # exp_result_dict, fidelity = run_xpath_explanation(args, get_fidelity=True, top_k=3)
    # print(exp_result_dict, fidelity)

    # exp_dict = {'relative_stocks_dict': relative_stocks_dict, 'score_dict': score_dict}
    # save_path = r'.\results'
    # with open(r'{}/{}.json'.format(save_path, args.explainer), 'w') as f:
    #     json.dump(exp_dict, f)
