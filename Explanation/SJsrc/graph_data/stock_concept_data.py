import numpy as np
import pandas as pd
import os


def read_graph_dict(market, relation_name, data_path):
    path = os.path.join(
        data_path, f"data/{market}/stock_concept_graphs/{relation_name}/edge.csv"
    )
    df = pd.read_csv(path)
    stock_concept_graph = {}
    for index, row in df.iterrows():
        stock, concept = row["stock"], row["concept"]
        if concept not in stock_concept_graph:
            stock_concept_graph[concept] = []
        stock_concept_graph[concept].append(stock)
    return stock_concept_graph


def get_full_connection_matrix(stock_concept_graph, stocks_index_dict):
    """
    stocks_index_dict: represents the intended stock index   e.g. {"SH60000":0,"SH60001":1,...,}
    return: relation_encoding: a N*N*K adjacent matrix
    """
    stocks_num = len(stocks_index_dict)
    secotr_As = []
    for i, sector in enumerate(stock_concept_graph):
        sector_stocks = stock_concept_graph[sector]
        sector_A = np.zeros((stocks_num, stocks_num))
        for stock1 in sector_stocks:
            if stock1 not in stocks_index_dict:
                continue
            for stock2 in sector_stocks:
                if stock2 not in stocks_index_dict:
                    continue
                if stock1 == stock2:
                    continue
                sector_A[stocks_index_dict[stock1], stocks_index_dict[stock2]] = 1
        if np.sum(sector_A) > 0:
            secotr_As.append(sector_A)
    secotr_As.append(np.eye(stocks_num))
    return np.stack(secotr_As, axis=2)
