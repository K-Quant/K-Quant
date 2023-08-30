import numpy as np
import pandas as pd
import os


def read_one_matrix(market, stocks_index_dict, relation_name, data_path):
    path = os.path.join(data_path, f"data/{market}/stock_stock_graphs/{relation_name}")
    stocks_num = len(stocks_index_dict)
    relation_A = np.zeros((stocks_num, stocks_num))
    if market == "A_share":
        temp = []
        quarters = os.listdir(path)
        for quarter in quarters:
            temp.append(pd.read_csv(os.path.join(path, quarter, "edge.csv")))
        relation_df = pd.concat(temp)
    else:
        relation_df = pd.read_csv(os.path.join(path, "edge.csv"))
    for index, row in relation_df.iterrows():
        stock1, stock2 = row["head_code"], row["tail_code"]
        if stock1 not in stocks_index_dict or stock2 not in stocks_index_dict:
            continue
        relation_A[stocks_index_dict[stock1]][stocks_index_dict[stock2]] = 1
    return relation_A


def get_all_matrix(market, stocks_index_dict, data_path):
    """
    stocks_index_dict: represents the intended stock index   e.g. {"SH60000":0,"SH60001":1,...,}
    return: relation_encoding: a N*N*K adjacent matrix
    """
    path = os.path.join(data_path, f"data/{market}/stock_stock_graphs/")
    stocks_num = len(stocks_index_dict)
    relation_As = []
    relation_types = os.listdir(path)
    for relation_name in relation_types:
        relation_A = read_one_matrix(
            market, stocks_index_dict, relation_name, data_path
        )
        if np.sum(relation_A) > 0:
            relation_As.append(relation_A)
    relation_As.append(np.eye(stocks_num))
    return np.stack(relation_As, axis=2)
