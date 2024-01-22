import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Explanation.utils import *
from Explanation.SJsrc import *
from Explanation.HKUSTsrc import *


class ParseConfigFile(argparse.Action):
    def __call__(self, parser, namespace, filename, option_string=None):
        if not os.path.exists(filename):
            raise ValueError("cannot find config at `%s`" % filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_name", default="NRSR")
    parser.add_argument("--model_path", default="..\parameter")
    parser.add_argument("--num_relation", type=int, default=102)
    parser.add_argument("--d_feat", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--loss_type", default="")
    parser.add_argument("--config", action=ParseConfigFile, default="")
    # for ts lib model
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--moving_avg", type=int, default=21)
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="b",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=False,
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--n_heads", type=int, default=1, help="num of heads")
    parser.add_argument("--d_ff", type=int, default=64, help="dimension of fcn")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--e_layers", type=int, default=8, help="num of encoder layers")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument(
        "--pred_len",
        type=int,
        default=-1,
        help="the length of pred squence, in regression set to -1",
    )
    parser.add_argument("--de_norm", default=True, help="de normalize or not")

    # data
    parser.add_argument("--data_set", type=str, default="csi360")
    parser.add_argument("--target", type=str, default="t+0")
    parser.add_argument("--pin_memory", action="store_false", default=True)
    parser.add_argument("--batch_size", type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument("--least_samples_num", type=float, default=1137.0)
    parser.add_argument("--label", default="")  # specify other labels
    parser.add_argument("--start_date", default="2019-01-01")
    parser.add_argument("--end_date", default="2019-01-05")

    # input for csi 300
    parser.add_argument("--data_root", default="..\Data")
    parser.add_argument(
        "--market_value_path", default="..\Data\csi300_market_value_07to20.pkl"
    )
    parser.add_argument(
        "--stock2concept_matrix", default="..\Data\csi300_stock2concept.npy"
    )
    parser.add_argument(
        "--stock2stock_matrix", default="..\Data\csi300_multi_stock2stock_all.npy"
    )
    parser.add_argument("--stock_index", default="..\Data\csi300_stock_index.npy")
    parser.add_argument("--model_dir", default="..\parameter")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--device", default="cpu")

    # relation
    parser.add_argument(
        "--relation_name_list_file", default=r"..\Data\relation_name_list.json"
    )

    args = parser.parse_args()

    return args


def run_explanation(args):
    if args.explainer == "xpathExplainer":

        def select_explanation(args, exp_dict):
            new_exp_dict = {}
            for date in args.date_list:
                new_exp_dict[date] = {}
                exp = exp_dict[date]
                for stock in args.stock_list:
                    if stock not in exp.keys():
                        print(r"No stock {}!".format(stock))
                    else:
                        new_exp_dict[date][stock] = []
                        for re_stocks, score in exp[stock].items():
                            re_s = re_stocks
                            new_exp_dict[date][stock] += [(re_s, score)]
            return new_exp_dict

        data_df = load_data_df(args)
        model = get_model(args)
        explainer = xPath(graph_model="heterograph", num_layers=1, device=device)
        explanation, score_dict, scores_dict = model.get_explanation(
            data_df, explainer, None, step_size=1, top_k=args.top_k
        )

        # relative_stocks_dict = select_explanation(args, explanation)
        relative_stocks_dict = select_top_k_related_stock(explanation, k=args.top_k)

        return relative_stocks_dict, scores_dict

    else:
        data_loader = create_data_loaders(args)
        explanation = Explanation(args, data_loader, explainer_name=args.explainer)
        explanation.explain()
        relative_stocks_dict = explanation.check_all_relative_stock()
        score_dict = explanation.check_all_assessment_score()
        return relative_stocks_dict, score_dict


def run_input_gradient_explanation(args):
    param_dict = json.load(
        open(args.model_path + "/" + args.model_name + "/info.json")
    )["config"]
    # The param_dict is really confusing, so I add the following lines to make it work at my computer.
    param_dict["market_value_path"] = args.market_value_path
    param_dict["stock2stock_matrix"] = args.stock2stock_matrix
    param_dict["stock_index"] = args.stock_index
    param_dict["model_dir"] = args.model_dir + "/" + args.model_name
    param_dict["data_root"] = args.data_root
    param_dict["start_date"] = args.start_date
    param_dict["end_date"] = args.end_date
    param_dict["device"] = device
    param_dict["graph_data_path"] = param_dict["stock2stock_matrix"]
    param_dict["graph_model"] = param_dict["model_name"]
    param_args = argparse.Namespace(**param_dict)
    if not args.explainer == "inputGradientExplainer":
        return None
    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
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

    data_loader = create_data_loaders(args)
    explanation = Explanation(param_args, data_loader, explainer_name=args.explainer)
    with open(args.relation_name_list_file, "r") as json_file:
        _relation_name_list = json.load(json_file)
    res = explanation.explain_xpath(
        stock_list=args.stock_list,
        get_fidelity=get_fidelity,
        top_k=top_k,
        relation_list=_relation_name_list,
    )
    return res


def check_all_relative_stock(args, exp_result_dict):
    _stock_index = np.load(args.stock_index, allow_pickle=True).item()
    with open(args.relation_name_list_file, "r") as json_file:
        _relation_name_list = json.load(json_file)
    index_to_stock_id = {index: stock_id for stock_id, index in _stock_index.items()}

    relative_stocks_dict = {}
    for date in exp_result_dict.keys():
        exp_graph = exp_result_dict[date]["expl_graph"]
        stock_index_in_adj = exp_result_dict[date]["stock_index_in_adj"]
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
                relative_scores = {
                    rel_name: score
                    for rel_name, score in zip(_relation_name_list, scores)
                    if score != 0
                }

                if relative_scores:
                    related_stocks.append(
                        {
                            "relative_stock_id": other_stock_id,
                            "total_score": total_score,
                            "individual_scores": relative_scores,
                        }
                    )

            # Sort related_stocks based on total_score
            related_stocks.sort(key=lambda x: x["total_score"], reverse=True)
            # Keep only the top three related stocks
            top_three_related_stocks = related_stocks[:3]

            for entry in top_three_related_stocks:
                other_stock_id = entry["relative_stock_id"]
                relative_stocks_dict[date][stock_id][other_stock_id] = {
                    "total_score": entry["total_score"],
                    "individual_scores": entry["individual_scores"],
                }

    return relative_stocks_dict


def evaluate_fidelity(explanation, exp_result_dict, p=0.2):
    if exp_result_dict is None:
        print("exp_result_dict is None")
    evaluation_results = {}
    for date, exp in exp_result_dict.items():
        expl_graph = exp["expl_graph"]
        origin_graph = exp["origin_graph"]
        feature = exp["feature"]
        fidelity = explanation.evaluate_explanation(
            feature, expl_graph, origin_graph, p=p
        )
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
                    sorted_rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)[
                        :_k
                    ]
                    stock_dict[os] = sorted_rs
            else:
                sorted_rs = sorted(rs.items(), key=lambda x: x[1], reverse=True)[:k]
                stock_dict[os] = sorted_rs
        new_relative_stocks_dict[d] = stock_dict
    return new_relative_stocks_dict


def get_results(
    args, start_date, end_date, explainer, check_stock_list, check_date_list
):
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
            print(
                r"--------------------------------------------------------------------"
            )
            print(r"股票 {} 解释结果如下：".format(stock))
            print(r"最相关的股票与得分{}".format(relative_stocks_dict[date][stock]))
            print(r"对该解释结果的评价如下：")
            print(
                r"总得分：{}，保真度得分：{}，准确性得分：{}， 稀疏性得分：{}".format(
                    score["score"], score["f_score"], score["a_score"], score["s_score"]
                )
            )
def convert_float32_to_float(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_float32_to_float(value)
    elif isinstance(data, np.float32):
        data = float(data)
    return data

if __name__ == "__main__":
    args = parse_args()
    args.start_date = "2022-06-01"
    args.end_date = "2023-06-30"
    args.stock_list = ["SH600000"]
    args.model_name = "relation_GATs"
    # args.date_list = ['2022-06-02']
    stock_list = [
        "SH600000",
        "SH600009",
        "SH600010",
        "SH600011",
        "SH600015",
        "SH600016",
        "SH600018",
        "SH600019",
        "SH600025",
        "SH600028",
        "SH600029",
        "SH600030",
        "SH600031",
        "SH600036",
        "SH600048",
        "SH600050",
        "SH600061",
        "SH600079",
        "SH600085",
        "SH600104",
        "SH600109",
        "SH600111",
        "SH600115",
        "SH600132",
        "SH600143",
        "SH600150",
        "SH600161",
        "SH600176",
        "SH600183",
        "SH600196",
        "SH600276",
        "SH600309",
        "SH600332",
        "SH600346",
        "SH600352",
        "SH600362",
        "SH600383",
        "SH600406",
        "SH600426",
        "SH600436",
        "SH600438",
        "SH600489",
        "SH600519",
        "SH600547",
        "SH600570",
        # "SZ002958",
        "SH600585",
        "SH600588",
        "SH600600",
        "SH600606",
        "SH600655",
        "SH600660",
        "SH600690",
        "SH600741",
        "SH600760",
        "SH600795",
        "SH600809",
        "SH600837",
        "SH600848",
        "SH600886",
        "SH600887",
        "SH600893",
        "SH600900",
        "SH600919",
        "SH600926",
        "SH600958",
        "SH600989",
        "SH600999",
        "SH601006",
        "SH601009",
        "SH601012",
        "SH601021",
        "SH601066",
        "SH601088",
        "SH601108",
        "SH601111",
        "SH601138",
        "SH601155",
        "SH601162",
        "SH601166",
        "SH601169",
        "SH601186",
        "SH601211",
        "SH601216",
        "SH601225",
        "SH601229",
        "SH601231",
        "SH601236",
        "SH601238",
        "SH601288",
        "SH601318",
        "SH601319",
        "SH601328",
        "SH601336",
        "SH601360",
        "SH601377",
        "SH601390",
        "SH601398",
        "SH601600",
        "SH601601",
        "SH601607",
        "SH601618",
        "SH601628",
        "SH601633",
        "SH601668",
        "SH601669",
        "SH601688",
        "SH601698",
        "SH601766",
        "SH601788",
        "SH601800",
        "SH601808",
        "SH601818",
        "SH601838",
        "SH601857",
        "SH601877",
        "SH601878",
        "SH601881",
        "SH601888",
        "SH601898",
        "SH601899",
        "SH601901",
        "SH601919",
        "SH601933",
        "SH601939",
        "SH601966",
        "SH601985",
        "SH601988",
        "SH601989",
        "SH601998",
        "SH603019",
        "SH603160",
        "SH603259",
        "SH603260",
        "SH603288",
        "SH603501",
        "SH603799",
        "SH603833",
        "SH603899",
        "SH603986",
        "SH603993",
        "SZ000001",
        "SZ000002",
        "SZ000063",
        "SZ000066",
        "SZ000069",
        "SZ000100",
        "SZ000157",
        "SZ000166",
        "SZ000301",
        "SZ000333",
        "SZ000338",
        "SZ000425",
        "SZ000538",
        "SZ000568",
        "SZ000596",
        "SZ000625",
        "SZ000651",
        "SZ000661",
        "SZ000703",
        "SZ000708",
        "SZ000725",
        "SZ000768",
        "SZ000776",
        "SZ000783",
        "SZ000786",
        "SZ000800",
        "SZ000858",
        "SZ000876",
        "SZ000895",
        "SZ000938",
        "SZ000963",
        "SZ000977",
        "SZ001979",
        "SZ002001",
        "SZ002007",
        "SZ002008",
        "SZ002024",
        "SZ002027",
        "SZ002032",
        "SZ002044",
        "SZ002049",
        "SZ002050",
        "SZ002120",
        "SZ002129",
        "SZ002142",
        "SZ002179",
        "SZ002202",
        "SZ002230",
        "SZ002236",
        "SZ002241",
        "SZ002252",
        "SZ002271",
        "SZ002304",
        "SZ002311",
        "SZ002352",
        "SZ002410",
        "SZ002415",
        "SZ002460",
        "SZ002466",
        "SZ002475",
        "SZ002493",
        "SZ002555",
        "SZ002568",
        "SZ002594",
        "SZ002601",
        "SZ002602",
        "SZ002607",
        "SZ002624",
        "SZ002714",
        "SZ002736",
        "SZ002841",
        "SZ002916",
        "SZ002938",
        "SZ300003",
        "SZ300015",
        "SZ300033",
        "SZ300059",
        "SZ300122",
        "SZ300124",
        "SZ300142",
        "SZ300144",
        "SZ300347",
        "SZ300408",
        "SZ300413",
        "SZ300433",
        "SZ300498",
    ]


     # for inputGradient:
    
    # args.explainer = "inputGradientExplainer"
    # exp_result_dict, explanation = run_input_gradient_explanation(args)

    # list = ["SH600000"]
    # for stock in stock_list:
    #     args.stock_list = [stock]

    #     exp_result_dict_sorted = search_stocks_prediction_explanation(
    #         args, exp_result_dict
    #     )
    #     exp_json_data = convert_float32_to_float(exp_result_dict_sorted)
    #     print(exp_json_data)

    #     with open("./outputDataGATs/" + stock + ".json", "w") as f:
    #         json.dump(exp_json_data, f)

    # for xpath:
    args.explainer = 'xpathExplainer'
    args.stock_list = stock_list
    exp_result_dict = run_xpath_explanation(args, get_fidelity=False, top_k=3)
    print(exp_result_dict)
    with open("xpath_GATs.json", "w") as f:
        json.dump(exp_result_dict, f)

    # exp_result_dict = search_stocks_prediction_explanation(args, exp_result_dict)
    # # fidelity = evaluate_fidelity(explanation, exp_result_dict, 0.2)
    # print(exp_result_dict)
        
    # exp_result_dict, fidelity = run_xpath_explanation(args, get_fidelity=True, top_k=3)
    # print(exp_result_dict, fidelity)

    # exp_dict = {'relative_stocks_dict': relative_stocks_dict, 'score_dict': score_dict}
    # save_path = r'.\results'
    # with open(r'{}/{}.json'.format(save_path, args.explainer), 'w') as f:
    #     json.dump(exp_dict, f)
