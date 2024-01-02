import argparse

import numpy as np
from sklearn.metrics import ndcg_score

from Explanation.ExplanationInterface import evaluate_fidelity, check_all_relative_stock
from Explanation.HKUSTsrc import Explanation


def cal_assessment(param_dict, data_loader, model, device):
    from run_assessment import predict

    preds = predict(param_dict, data_loader, model, device)

    if param_dict['model_name'] == 'NRSR' or 'relation_GATs':
        explainable = cal_explainable(param_dict, data_loader, device)
    else:
        explainable = 0

    reliability = cal_reliability(preds)
    stability = cal_stability(preds)
    robustness = cal_robustness(preds, param_dict, data_loader, model, device)
    transparency = cal_transparency(param_dict['model_name'])

    return reliability, stability, explainable, robustness, transparency


def cal_explainable(param_dict, data_loader, device, explainer='inputGradientExplainer', p=0.2):
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=device)
    parser.add_argument('--graph_data_path', default=param_dict['stock2stock_matrix'])
    parser.add_argument('--model_dir', type=str, default=param_dict['model_dir'])
    parser.add_argument('--graph_model', type=str, default=param_dict['model_name'])
    parser.add_argument('--stock_index', type=str, default=r"D:\ProjectCodes\K-Quant\Data\csi300_stock_index.npy")

    parser.add_argument('--d_feat', type=int, default=param_dict['d_feat'])
    parser.add_argument('--num_layers', type=int, default=param_dict['num_layers'])

    # 选择预测模型
    args = parser.parse_known_args()[0]
    explanation = Explanation(args, data_loader, explainer_name=explainer)
    exp_result_dict = explanation.explain()
    # check_all_relative_stock(args, exp_result_dict)
    evaluation_results = evaluate_fidelity(explanation, exp_result_dict, p=p)
    fidelity = np.mean(np.array(list(evaluation_results.values())))
    return fidelity



def cal_reliability(preds):
    preds = preds[~np.isnan(preds['label'])]
    rank_ic_mean = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return rank_ic_mean


def cal_stability(preds):
    preds = preds[~np.isnan(preds['label'])]
    rank_ic_std = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).std()
    return rank_ic_std


def cal_robustness(preds, param_dict, data_loader, model, device, r=2):
    from run_assessment import predict

    r_scores = []
    for i in range(r):
        perturb_preds = predict(param_dict, data_loader, model, device, noise=True)
        r_score = preds.score.corr(perturb_preds.score, method='spearman')
        r_scores.append(r_score)
    return np.mean(r_scores)


def cal_transparency(model_name):
    model_name = model_name.lower()
    if model_name in ['linear_reg', 'logistic_reg', 'GLM', 'knn',
                      'decision_tree', 'random_forest', 'gbdt', 'xgboost', 'lightgbm']:
        # fully transparent
        return 2
    elif model_name in ['relation_gats', 'hist', 'rsr', 'kenhance']:
        # knowledge based
        return 1
    else:
        # black box
        return 0


def cal_explainability(preds, model_name):
    reliability = cal_reliability(preds)
    # TODO cal fidelity
    fidelity = 1.0
    return fidelity * reliability


def np_relu(x):
    return x * (x > 0)
