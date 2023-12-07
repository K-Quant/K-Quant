import json

import numpy as np
import pandas as pd
# import numpy as np
import torch
from tqdm import tqdm

from Assessment.dataloader import create_data_loaders
from Model.model_pool.utils.utils import DotDict
from utils import get_model
from metrics import cal_reliability, cal_stability

time_series_library = [
    'DLinear',
    'Autoformer',
    'Crossformer',
    'ETSformer',
    'FEDformer',
    'FiLM',
    'Informer',
    'PatchTST'
]

relation_model_dict = [
    'NRSR',
    'relation_GATs_3heads',
    'KEnhance'
]


def set_model(args, param_dict, device):
    if param_dict['model_name'] == 'SFM':
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], output_dim=32, freq_dim=25,
                                                    hidden_size=param_dict['hidden_size'],
                                                    dropout_W=0.5, dropout_U=0.5, device=device)
    elif param_dict['model_name'] == 'ALSTM':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], param_dict['dropout'], 'LSTM')
    elif param_dict['model_name'] == 'Transformer':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], dropout=0.5)
    elif param_dict['model_name'] == 'NRSR':
        # the number of relations
        model = get_model(param_dict['model_name'])(num_relation=args.num_relation, d_feat=param_dict['d_feat'],
                                                    num_layers=param_dict['num_layers'])
    elif param_dict['model_name'] in time_series_library:
        model = get_model(param_dict['model_name'])(DotDict(param_dict))
    else:
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
    model.to(device)
    model.load_state_dict(torch.load(args.model_dir + "/" + args.model_name + '/model.bin', map_location=device))
    print('predict in ', param_dict['model_name'])
    return model


def cal_assessment(preds):
    reliability = cal_reliability(preds)
    stability = cal_stability(preds)
    return reliability, stability


def predict(param_dict, data_loader, model, device):
    model.eval()
    preds = []
    stock2stock_matrix = param_dict["stock2stock_matrix"]
    stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(device)
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            if param_dict['model_name'] == 'NRSR' or 'relation_GATs':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif param_dict['model_name'] in time_series_library:
                pred = model(feature, mask)
            else:
                pred = model(feature)

            preds.append(
                pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), },
                             index=index)
            )
    preds = pd.concat(preds, axis=0)
    return preds


def main(args):
    data_loader = create_data_loaders(args)
    param_dict = json.load(open(args.model_path + "/" + args.model_name + '/info.json'))['config']
    model = set_model(args, param_dict, args.device)
    preds = predict(param_dict, data_loader, model, args.device)
    reliability, stability = cal_assessment(preds)
    return reliability, stability

