import argparse
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from Explanation.HKUSTsrc import NRSR
from Model.model_pool.models.model import MLP, LSTM, GRU, GAT, ALSTM, KEnhance, relation_GATs


def get_model(model_name):
    a = model_name.upper()
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM
        # return LSTMModel

    if model_name.upper() == 'GRU':
        return GRU
        # return GRUModel
    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == "NRSR":
        return NRSR

    if model_name == "relation_GATs":
        return relation_GATs

    if model_name == "KEnhance":
        return KEnhance



def add_noise(x, level, noise_seed):
    torch.manual_seed(noise_seed)
    mu = 0
    sigma = level * torch.std(x)
    noise = torch.normal(mu, sigma, size=x.shape)
    hat_x = x + noise
    return hat_x


def transform_stock_code(stock_code):
    # 检查输入的字符串长度是否符合要求
    if len(stock_code) != 8 or not stock_code.startswith(("SH", "SZ")):
        raise ValueError("Invalid stock code format. Expected format: 'SHxxxxxx' or 'SZxxxxxx'.")

    # 提取股票代码和交易所代码
    exchange_code = stock_code[:2]
    stock_number = stock_code[2:]

    # 重新排列为 "xxxxxx.SH" 或 "xxxxxx.SZ" 的形式
    transformed_code = f"{stock_number}.{exchange_code}"

    return transformed_code


def normalize_results_list(results_list):
    # 转换为NumPy数组
    metrics_array = np.array(results_list)

    # 使用MinMaxScaler进行归一化
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(metrics_array)

    return normalized_metrics.tolist()


def normalize_assessment_results_list(assessment_results_list, num_selection = 5):
    if len(assessment_results_list) < num_selection:
        print(r"您做的选择数量需要大于{}种".format(str(num_selection)))
        return None
    else:
        # 提取所有的 evaluation_metrics_dict
        all_metrics = [result[1] for result in assessment_results_list]

        # 获取所有可能的得分项
        metric_names = list(all_metrics[0].keys())

        # 提取每个得分项的分数
        metrics_list_of_lists = [[metrics[metric] for metric in metric_names] for metrics in all_metrics]

        # 使用泛化的归一化函数
        normalized_metrics = normalize_results_list(metrics_list_of_lists)

        # 将归一化后的值更新回原列表
        for i, result in enumerate(assessment_results_list):
            for j, metric_name in enumerate(metric_names):
                result[1][metric_name] = normalized_metrics[i][j]

        return assessment_results_list


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
    parser.add_argument('--model_name', default='relation_GATs')
    parser.add_argument('--model_path', default='..\parameter')
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
    parser.add_argument('--data_root', default='..\Data')
    parser.add_argument('--market_value_path', default= '..\Data\csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='..\Data\csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='..\Data\csi300_multi_stock2stock_all.npy')
    parser.add_argument('--stock_index', default='..\Data\csi300_stock_index.npy')
    parser.add_argument('--model_dir', default='..\parameter')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    return args

