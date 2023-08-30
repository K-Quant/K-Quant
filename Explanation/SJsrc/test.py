import pandas as pd
import argparse
import torch
import pickle
import os
import numpy as np
import time
import logging
# from SJsrc import *
from  Explanation.SJsrc.graph_data import stock_stock_data, stock_concept_data
from  Explanation.SJsrc.models.ts_model import Graphs
from  Explanation.SJsrc.interpreter.attentionx import AttentionX
from  Explanation.SJsrc.interpreter.xpath import xPath
from  Explanation.SJsrc.interpreter.subgraphx import SubgraphXExplainer


def load_graph(args, market, relation_source, data_mix):
    indexs = data_mix.index.levels[1].tolist()
    indexs = list(set(indexs))
    stocks_sorted_list = sorted(indexs)
    print("number of stocks: ", len(stocks_sorted_list))
    stocks_index_dict = {}
    for i, stock in enumerate(stocks_sorted_list):
        stocks_index_dict[stock] = i
    n = len(stocks_index_dict.keys())
    if relation_source == 'stock-stock':
        rel_encoding = stock_stock_data.get_all_matrix(
            market, stocks_index_dict,
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/"
            data_path=args.data_root  # This is the graph_data path on the server
        )
        return rel_encoding, stocks_sorted_list
    elif relation_source == 'industry':
        industry_dict = stock_concept_data.read_graph_dict(
            market,
            relation_name="SW_belongs_to",
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/",
            data_path=args.data_root
        )
        return stock_concept_data.get_full_connection_matrix(
            industry_dict, stocks_index_dict
        ), stocks_sorted_list
    elif relation_source == 'full':
        return np.ones(shape=(n, n)), stocks_sorted_list
    else:
        raise ValueError("unknown graph name `%s`" % relation_source)


def init_logger(log_dir, args):
    os.makedirs(log_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    # log_file = log_root + f'/{args.graph_model}_{current_time}.log'
    # file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    # file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger('updateSecurity')
    logger.setLevel('INFO')
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_model(args):
    with open(
            f'D:\ProjectCodes\FintechRepo\data\dataframe_mix_csi300_rankTrue_alpha360_horizon1.pkl',
            'rb') as f:
        data_mix = pickle.load(f)
    rel_encoding, stock_name_list = load_graph(args, args.market, args.relation_type, data_mix['data'])
    model = Graphs(graph_model=args.graph_model,  # 'GAT' or 'simpleHGN', 'RSR'
                   d_feat=6, hidden_size=64, num_layers=1, loss="mse", dropout=0.7, n_epochs=100,
                   metric="loss", base_model="LSTM", use_residual=True, GPU=0, lr=1e-4,
                   early_stop=10, rel_encoding=rel_encoding, stock_name_list=stock_name_list,
                   num_graph_layer=1, logger=init_logger('../log', args))
    model.to(args.device)
    model_path = os.path.join(args.ckpt_root, f"{args.market}-{args.graph_model}-heterograph.pt")

    model.load_checkpoint(model_path)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Explanation evalation.")
    parser.add_argument("--data_root", type=str, default="D:\LearnCode\Fintech\interpreters-main", help="graph_data root path")
    parser.add_argument("--ckpt_root", type=str, default=r"D:\LearnCode\Fintech\interpreters-main\tmp_ckpt", help="ckpt root path")
    parser.add_argument("--result_root", type=str, default="/home/jiale/qlib_exp/results/",
                        help="explanation resluts root path")
    parser.add_argument("--market", type=str, default="A_share",
                        choices=["A_share"], help="market name")
    parser.add_argument("--relation_type", type=str, default="stock-stock",
                        choices=["stock-stock", "industry", "full"], help="relation type of graph")
    parser.add_argument("--graph_model", type=str, default="GAT",
                        choices=["RSR", "GAT", "simpleHGN"], help="graph moddel name")
    parser.add_argument("--graph_type", type=str, default="heterograph",
                        choices=["heterograph", "homograph"], help="graph type")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    args = parser.parse_args()
    return args


def create_loaders(start_date, end_date):
    """
    load qlib alpha360 data and split into train, validation and test loader
    :param args:
    :return:
    """
    # split those three dataset into train, valid and test
    with open(r'D:\LearnCode\Fintech\interpreters-main\data\alpha360.pkl', "rb") as fh:
        df_total = pickle.load(fh)

    slc = slice(pd.Timestamp(start_date), pd.Timestamp(end_date))
    df_check = df_total[slc]


    return df_check


def run_test(data_df):
    '''
    get explanations for stocks in a given time period
    '''
    # stocks = ['SH600000', 'SH600004', ]
    stocks = None
    start_time = '2017-01-02'
    end_time = '2017-01-10'

    xpath_explainer = xPath(graph_model=args.graph_type, num_layers=1, device=device)
    attn_explainer = AttentionX(graph_model=args.graph_type, num_layers=1, device=device)
    subagraphx_explainer = SubgraphXExplainer(graph_model=args.graph_type, num_layers=1, device=device)

    explanation = model.get_one_explanation(data_df, stocks, attn_explainer, top_k=3)
    return explanation





