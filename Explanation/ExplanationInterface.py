import json
from collections import Counter
from multiprocessing import Pool

from Explanation.utils import *
from Explanation.SJsrc import *
from Explanation.HKUSTsrc import *


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='NRSR')
    parser.add_argument('--model_path', default='D:\Research\Fintech\K-Quant\parameter')
    parser.add_argument('--num_relation', type= int, default=102)
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--loss_type', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    # for ts lib model
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--moving_avg', type=int, default=21)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='b',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=False)
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--pred_len', type=int, default=-1, help='the length of pred squence, in regression set to -1')
    parser.add_argument('--de_norm', default=True, help='de normalize or not')

    # data
    parser.add_argument('--data_set', type=str, default='csi360')
    parser.add_argument('--target', type=str, default='t+0')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--start_date', default='2019-01-01')
    parser.add_argument('--end_date', default='2019-01-05')

    # input for csi 300
    parser.add_argument('--data_root', default='D:\Research\Fintech\K-Quant\Data')
    parser.add_argument('--market_value_path', default= 'D:\Research\Fintech\K-Quant\Data\csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='D:\ProjectCodes\K-Quant\Data\csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='D:\Research\Fintech\K-Quant\Data\csi300_multi_stock2stock_hidy_2024.npy')


    parser.add_argument('--stock_index', default='D:\Research\Fintech\K-Quant\Data\csi300_stock_index_2024.npy')
    parser.add_argument('--model_dir', default='D:\Research\Fintech\K-Quant\parameter')
    parser.add_argument('--events_files', default='D:\Research\Fintech\K-Quant\Data\event_data_sigle_stock.json')

    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')

    # relation
    parser.add_argument('--relation_name_list_file', default=r'D:\Research\Fintech\K-Quant\Data\new_relation_name_list2024.json')

    args = parser.parse_args()

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


def load_params(args):
    param_dict = json.load(open(args.model_path + "/" + args.model_name + '/info.json'))['config']
    # The param_dict is really confusing, so I add the following lines to make it work at my computer.
    param_dict['market_value_path'] = args.market_value_path
    param_dict['stock2stock_matrix'] = args.stock2stock_matrix
    param_dict['stock_index'] = args.stock_index
    param_dict['model_dir'] = args.model_dir + "/" + args.model_name
    param_dict['data_root'] = args.data_root
    param_dict['start_date'] = args.start_date
    param_dict['end_date'] = args.end_date
    param_dict['device'] = device
    param_dict['graph_data_path'] = param_dict['stock2stock_matrix']
    param_dict['graph_model'] = param_dict['model_name']
    param_args = argparse.Namespace(**param_dict)
    return param_args


def run_input_gradient_explanation(args, event_data):
    param_args = load_params(args)
    if not args.explainer == 'inputGradientExplainer':
        return None
    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
    exp_result_dict = explanation.explain()
    exp_result_dict, sorted_stock_rank = check_all_relative_stock(args, exp_result_dict, event_data)

    return exp_result_dict, sorted_stock_rank, explanation


def get_previous_days(start_date, n):
    from datetime import datetime, timedelta
    # 将字符串日期转换为 datetime 对象
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    # 创建一个空列表，用于存储检索到的日期
    date_list = []

    # 从指定日期开始向前检索 n 天
    for i in range(n):
        # 计算当前日期减去 i 天后的日期
        previous_date = start_date - timedelta(days=i)
        # 将日期添加到列表中
        date_list.append(previous_date.strftime("%Y-%m-%d"))

    return date_list



def convert_stock_code(stock_code):
    # 提取交易所代码和数字部分
    exchange_code = stock_code[:2]
    numeric_part = stock_code[2:]

    # 将交易所代码移动到末尾
    converted_stock_code = numeric_part + '.' + exchange_code

    return converted_stock_code


def get_events(date, num_days, stock_id, event_data):
    converted_stock_id = convert_stock_code(stock_id)
    previous_days = get_previous_days(date, num_days)
    events = {}

    if converted_stock_id in event_data.keys():
        have_event_dates = [d for d in event_data[converted_stock_id].keys() if d in previous_days]
        for d in have_event_dates:
            events[d] = event_data[converted_stock_id][d]
        return events

    else:
        return {}


def search_stocks_prediction_explanation(args, exp_result_dict, event_data):
    result_subset = {}
    for stock_id in args.stock_list:
        result_subset[stock_id] = {}
        for date, result_data in exp_result_dict.items():
            # Check if date is within the specified range
            if args.start_date <= date <= args.end_date:
                result_subset[stock_id][date] = {}
                for key, res in result_data[stock_id].items():
                    result_subset[stock_id][date][key] = {
                        "explanation": res,
                        "events": get_events(date, args.seq_len, key, event_data)
                    }
    return result_subset


def check_stocks_with_event(args, exp_result_dict, event_data):
    relative_stocks_dict = {}
    for date, v in exp_result_dict.items():
        relative_stocks_dict[date] = {}
        for stock_id, va in v.items():

            relative_stocks_dict[date][stock_id] = {}
            for other_stock_id, res in va.items():
                if other_stock_id == 'pred_result':
                    continue
                relative_stocks_dict[date][stock_id][other_stock_id] ={
                    'total_score': res['score'],
                    'relations': res['relations'],
                    'events': get_events(date, args.seq_len, other_stock_id, event_data)
                }

    return relative_stocks_dict


def run_xpath_explanation(args, event_data, get_fidelity=False, top_k=3):
    param_args = load_params(args)
    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
    with open(args.relation_name_list_file, 'r') as json_file:
        _relation_name_list = json.load(json_file)
    res, stock_rank = explanation.explain_x(stock_list=args.check_stock_list, get_fidelity=get_fidelity,
                                         top_k=top_k, relation_list=_relation_name_list)

    res = check_stocks_with_event(args, res, event_data)
    return res, stock_rank


def run_gnn_explainer(args, event_data, top_k=3):
    param_args = load_params(args)
    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
    with open(args.relation_name_list_file, 'r') as json_file:
        _relation_name_list = json.load(json_file)
    res, stock_rank = explanation.explain_x(stock_list=args.check_stock_list, top_k=top_k,
                                            relation_list=_relation_name_list)
    res = check_stocks_with_event(args, res, event_data)
    return res, stock_rank


def run_hencex_explainer(args, event_data, top_k=3):
    param_args = load_params(args)
    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
    with open(args.relation_name_list_file, 'r') as json_file:
        _relation_name_list = json.load(json_file)
    res, stock_rank = explanation.explain_x(stock_list=args.check_stock_list, top_k=top_k,
                                            relation_list=_relation_name_list)
    res = check_stocks_with_event(args, res, event_data)
    return res, stock_rank


def check_all_relative_stock(args, exp_result_dict, event_data):
    _stock_index = np.load(args.stock_index, allow_pickle=True).item()
    with open(args.relation_name_list_file, 'r',encoding='utf-8') as json_file:
        _relation_name_list = json.load(json_file)
    index_to_stock_id = {index: stock_id for stock_id, index in _stock_index.items()}

    relative_stocks_dict = {}
    stock_rank = {}
    for date in exp_result_dict.keys():
        pred = exp_result_dict[date]['pred']
        exp_graph = exp_result_dict[date]['expl_graph']
        stock_index_in_adj = exp_result_dict[date]['stock_index_in_adj']
        relative_stocks_dict[date] = {}
        stock_rank[date] = {}

        num_stocks, _, num_relations = exp_graph.shape

        for i in range(num_stocks):
            stock_id = index_to_stock_id[stock_index_in_adj[i]]  # 获取股票编号
            relative_stocks_dict[date][stock_id] = {}
            related_stocks = []
            pred_result = pred[i]
            stock_rank[date][stock_id] = pred_result
            # if stock_id not in args.check_stock_list:
            #     continue
            relative_stocks_dict[date][stock_id]['pred_result'] = pred_result
            for j in range(num_stocks):
                other_stock_id = index_to_stock_id[stock_index_in_adj[j]]
                scores = exp_graph[i, j, :]
                total_score = scores.sum()
                relative_scores = [rel_name for rel_name, score in zip(_relation_name_list, scores) if score != 0]

                if relative_scores:
                    related_stocks.append({
                        'relative_stock_id': other_stock_id,
                        'total_score': total_score,
                        'relation': relative_scores
                    })

            # Sort related_stocks based on total_score
            related_stocks.sort(key=lambda x: x['total_score'], reverse=True)
            # Keep only the top three related stocks
            top_three_related_stocks = related_stocks[:3]

            for entry in top_three_related_stocks:
                other_stock_id = entry['relative_stock_id']
                relative_stocks_dict[date][stock_id][other_stock_id] = {
                    'total_score': entry['total_score'],
                    'relation': entry['relation'],
                    'events': get_events(date, args.seq_len, other_stock_id, event_data)
                }
        stock_rank[date] = dict(sorted(stock_rank[date].items(), key=lambda item: item[1], reverse=True))
        stock_rank[date] = stock_rank[date].keys()
    return relative_stocks_dict, stock_rank


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
    #
    # pool = Pool(10)
    #
    #
    # processed_n_data = pool.map(tensorize_with_label, zip(smiles, labels))

    args = parse_args()
    args.start_date = '2022-01-10'
    args.end_date = '2022-01-11'
    # args.check_stock_list = ['SH600383']
    args.model_name =  'NRSR'
    events_files = args.events_files
    # args.date_list = ['2022-06-02']
    with open(events_files, 'r', encoding='utf-8') as f:
        events_data = json.load(f)

    args.explainer = 'inputGradientExplainer'
    exp_result_dict, sorted_stock_rank, explanation = run_input_gradient_explanation(args, events_data)
    # fidelity = evaluate_fidelity(explanation, exp_result_dict, 0.2)
    with open(r"D:\Research\Fintech\K-Quant\test.json", 'w', encoding='utf-8') as f:
        json.dump(exp_result_dict, f)
    print(exp_result_dict)

    # for xpath:
    # for inputGradient:
    # args.explainer = 'xpathExplainer'
    # exp_result_dict, sorted_stock_rank = run_xpath_explanation(args, events_data, get_fidelity=False, top_k=3)
    # print(sorted_stock_rank)

    # print(exp_result_dict)
    # exp_result_dict, fidelity = run_xpath_explanation(args, get_fidelity=True, top_k=3)
    # print(exp_result_dict, fidelity)

    # for GNNExplainer:
    # args.explainer = 'gnnExplainer'
    # exp_result_dict, sorted_stock_rank = run_gnn_explainer(args, events_data, top_k=3)
    # print(exp_result_dict)

    # for EffectExplainer:
    # args.explainer = 'hencexExplainer'
    # exp_result_dict, sorted_stock_rank = run_hencex_explainer(args, events_data, top_k=3)
    # print(exp_result_dict)

    # print stock names
    # import pickle
    # stock_2_name = pickle.load(open(r'../Data/company_full_name.pkl', 'rb'))
    # for date in exp_result_dict.keys():
    #     stock_names = []
    #     for target_stock in exp_result_dict[date].keys():
    #         for stock in exp_result_dict[date][target_stock].keys():
    #             if stock in stock_2_name.keys():
    #                 stock_names.append(stock_2_name[stock])
    #         print(f'{date} {stock_2_name[target_stock]}: {stock_names}')

    # save the explanation result
    # with open(f'../results/{args.explainer}_{args.model_name}.json', 'w') as f:
    #     json.dump(exp_result_dict, f)
