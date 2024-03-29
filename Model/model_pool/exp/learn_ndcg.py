"""
different with classical single-step stock forecasting task
we treat stock forecasting as a multi-classification approx ndcg task
we will devide stocks into 4 different class according to their ranking in a single day
and use approxNDCG as the loss function and use NDCG as the evaluation while training
the basic network is like multi-classification
"""
import torch
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, sys.path[0]+"/../")
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR, relation_GATs, relation_GATs_3heads
from qlib.contrib.model.pytorch_transformer import Transformer
from models.DLinear import DLinear_model
from models.Autoformer import Model as autoformer
from models.Crossformer import Model as crossformer
from models.ETSformer import Model as ETSformer
from models.FEDformer import Model as FEDformer
from models.FiLM import Model as FiLM
from models.Informer import Model as Informer
from models.PatchTST import Model as PatchTST
from utils.utils import metric_fn, mse, loss_ic, pair_wise_loss, NDCG_loss, ApproxNDCG_loss, cross_entropy, \
    generate_label, evaluate_mc, class_approxNDCG, softclass_NDCG, NDCG_evaluation
from utils.dataloader import create_loaders
import warnings
import logging


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12
warnings.filterwarnings('ignore')

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
    'RSR',
    'relation_GATs',
    'relation_GATs_3heads'
]


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM
        # return LSTMModel

    if model_name.upper() == 'GRU':
        return GRU
        # return GRUModel

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'SFM':
        return SFM

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'RELATION_GATS':
        return relation_GATs

    if model_name.upper() == 'RELATION_GATS_3HEADS':
        return relation_GATs_3heads

    if model_name.upper() == 'DLINEAR':
        return DLinear_model

    if model_name.upper() == 'AUTOFORMER':
        return autoformer

    if model_name.upper() == 'CROSSFORMER':
        return crossformer

    if model_name.upper() == 'ETSFORMER':
        return ETSformer

    if model_name.upper() == 'FEDFORMER':
        return FEDformer

    if model_name.upper() == 'FILM':
        return FiLM

    if model_name.upper() == 'INFORMER':
        return Informer

    if model_name.upper() == 'PATCHTST':
        return PatchTST

    raise ValueError('unknown model name `%s`'%model_name)


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


global_log_file = None


def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1


def train_epoch(epoch, model, optimizer, train_loader, writer, args,
                stock2concept_matrix=None, stock2stock_matrix = None):
    """
    train epoch function
    :param epoch: number of epoch of training
    :param model: model that will be used
    :param optimizer:
    :param train_loader:
    :param writer:
    :param args:
    :param stock2concept_matrix:
    :return:
    """

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value, stock_index, _, mask = train_loader.get(slc)
        # we get feature and label, pred in classification is a tensor, first
        # 将其变为分类之后，给不同的类别不同的weight，算approxNDCG，如何回传梯度？ -- 可以使用软分类方法
        # 参考传统LTR数据的处理方法
        if args.model_name == 'HIST':
            # if HIST is used, take the stock2concept_matrix and market_value
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            # the stock2concept_matrix[stock_index] is a matrix, shape is (the number of stock index, predefined con)
            # the stock2concept_matrix has been sort to the order of stock index
        elif args.model_name in relation_model_dict:
            pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
        elif args.model_name in time_series_library:
            # new added
            pred = model(feature, mask)
        else:
            # other model only use feature as input
            # for multi-class, here we get a [B, N]
            pred = model(feature)

        loss = class_approxNDCG(pred, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index, mask = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name in relation_model_dict:
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif args.model_name in time_series_library:
                # new added
                pred = model(feature, mask)
            else:
                pred = model(feature)

            loss = class_approxNDCG(pred, label)
            pred_label, true_label = softclass_NDCG(pred, label)
            # here pred and ground truth are real numbers from 0~3
            preds.append(pd.DataFrame({'pred': pred_label.cpu().numpy(),
                                       'ground_truth': true_label.cpu().numpy().astype(float),}, index=index))

            # preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())

    # evaluate
    preds = pd.concat(preds, axis=0)
    # use metric_fn to compute precision, recall, ic and rank ic
    ndcg = NDCG_evaluation(preds)

    '''
    here change scores to others 
    score is also very important, since it decide which model to choose 
    '''
    # scores = ic + ndcg[100] + precision[100]
    scores = -np.mean(losses)  # score is the opposite loss
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, ndcg


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index, mask = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name in relation_model_dict:
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif args.model_name in time_series_library:
                # new added
                pred = model(feature, mask)
            else:
                pred = model(feature)
            pred_label, true_label = softclass_NDCG(pred,label)
            preds.append(pd.DataFrame({'pred': pred_label.cpu().numpy(),
                                       'ground_truth': true_label.cpu().numpy().astype(float),}, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = '../ouput/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args, device=device)

    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    if args.model_name == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    if args.model_name in relation_model_dict:
        stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
        num_relation = stock2stock_matrix.shape[2]


    all_ndcg = []
    for times in range(args.repeat):
        pprint('create model...')
        if args.model_name == 'SFM':
            model = get_model(args.model_name)(d_feat=args.d_feat, output_dim=32, freq_dim=25, hidden_size=args.hidden_size, dropout_W=0.5, dropout_U=0.5, device=device)
        elif args.model_name == 'ALSTM':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
        elif args.model_name == 'Transformer':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
        elif args.model_name == 'HIST':
            model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
        elif args.model_name == 'RSR':
            model = get_model(args.model_name)(num_relation=num_relation, d_feat=args.d_feat, num_layers=args.num_layers)
        elif args.model_name in time_series_library:
            model = get_model(args.model_name)(args)
        else:
            model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)
        for epoch in range(args.n_epochs):
            pprint('Running', times, 'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix)
            # save model  after every epoch
            # -------------------------------------------------------------------------
            # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
            # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            # when evaluating, use the avg_params in the current time to evaluate
            model.load_state_dict(avg_params)

            pprint('evaluating...')
            # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
            train_loss, train_score, train_ndcg = \
                test_epoch(epoch, model, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Train')
            val_loss, val_score, val_ndcg = \
                test_epoch(epoch, model, valid_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Valid')
            test_loss, test_score, test_ndcg = \
                test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            # score equals to ic here
            # pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            pprint('train_ndcg %.6f, valid_ndcg %.6f, test_ndcg %.6f'%(train_ndcg, val_ndcg, test_ndcg))

            # load back the current parameters
            model.load_state_dict(params_ckpt)

            if val_score > best_score:
                # the model performance is increasing
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                # the model performance is not increasing
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin')

        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            # do prediction on train, valid and test data
            pred = inference(model, eval(name+'_loader'), stock2concept_matrix=stock2concept_matrix,
                            stock2stock_matrix=stock2stock_matrix)
            ndcg = NDCG_evaluation(pred)
            pprint(name, ': NDCG ', ndcg)

            res[name + '-NDCG'] = ndcg

            # pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))
        all_ndcg.append(ndcg)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint('NDCG: %.4f (%.4f)' % (np.array(all_ndcg).mean(), np.array(all_ndcg).std()))

    pprint('finished.')


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
    parser.add_argument('--model_name', default='PatchTST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=1)
    # parser.add_argument('--loss_type', default='cross entropy')
    # also works for NDCG task
    parser.add_argument('--num_class', default=4, help='the number of class of stock sequence')

    # for ts lib model
    parser.add_argument('--task_name', type=str, default='multi-class', help='task setup')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--moving_avg', type=int, default=21)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
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
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--pred_len', type=int, default=-1, help='the length of pred squence, in regression set to -1')
    parser.add_argument('--de_norm', default=True, help='de normalize or not')

    # for REVin normalize
    parser.add_argument('--revin', default=False, help='use RevIn or not')
    parser.add_argument('--affine', default=True, help='use learnable parameters or not in RevIn')
    parser.add_argument('--subtract_last', default=False, help='subtract_last or not in RevIn')

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--smooth_steps', type=int, default=1)
    parser.add_argument('--metric', default='negative cross entropy')
    # parser.add_argument('--loss', default='cross entropy')
    parser.add_argument('--repeat', type=int, default=10)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--target', type=str, default='t+1')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--train_start_date', default='2008-01-01')
    parser.add_argument('--train_end_date', default='2018-12-31')
    parser.add_argument('--valid_start_date', default='2019-01-01')
    parser.add_argument('--valid_end_date', default='2020-12-31')
    parser.add_argument('--test_start_date', default='2021-01-01')
    parser.add_argument('--test_end_date', default='2022-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='PatchTST')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to22.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='./data/csi300_multi_stock2stock_all.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')
    parser.add_argument('--outdir', default='./output/ndcg/PatchTST_NDCG')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    for prediction, maybe repeat 5, early_stop 10 is enough
    """
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
