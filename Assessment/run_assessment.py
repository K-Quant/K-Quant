import json

# import numpy as np
import torch

from Assessment.dataloader import create_data_loaders
from Model.model_pool.utils.utils import DotDict
from utils import get_model

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
    'relation_GATs',
    'relation_GATs_3heads',
    'KEnhance'
]


def set_model(args, device):
    param_dict = json.load(open(args.model_path + "/" + args.model_name + '/info.json'))['config']
    stock2stock_matrix = param_dict['stock2stock_matrix']
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
    elif param_dict['model_name'] in relation_model_dict:  # the number of relations
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


def cal_assessment():
    pass


def main(args):
    data_loader = create_data_loaders(args)

    set_model(args, args.device)




