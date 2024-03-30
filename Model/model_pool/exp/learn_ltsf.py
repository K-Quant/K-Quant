"""
classical single-step stock forecasting task
follow HIST repo
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
from models.PatchTST import Model as PatchTST
from utils.utils import metric_fn, mse, loss_ic, pair_wise_loss, NDCG_loss, ApproxNDCG_loss
from utils.ltsf_dataloader import create_ltsf_loader
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


def get_model(model_name):

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


def train_epoch(model, optimizer, train_loader, args):
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
    mse_loss = torch.nn.MSELoss()
    for i, slc in tqdm(train_loader.iter_daily(), total=train_loader.daily_length):
        global_step += 1
        batch_x, batch_y, stock_index, mask = train_loader.get(slc)
        # feature here is the data batch_size, 360
        if args.output_attention:
            outputs = model(batch_x, None)[0]
        else:
            outputs = model(batch_x, None)
        '''
        in this task volume and amount have far lager scalar than others so we need to normalize both pred seq and 
        ground truth seq to insure that different scalar won't influence the mse error computing
        '''
        # normalize batch y
        batch_y = batch_y.reshape(len(batch_y), args.d_feat, -1)
        batch_y = batch_y.permute(0, 2, 1)
        means = batch_y.mean(1, keepdim=True).detach()
        batch_y = batch_y - means
        stdev = torch.sqrt(torch.var(batch_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_y /= stdev

        f_dim = 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:]
        loss = mse_loss(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """

    model.eval()
    mse_loss = torch.nn.MSELoss()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        batch_x, batch_y, index, _ = test_loader.get(slc)

        with torch.no_grad():
            if args.output_attention:
                outputs = model(batch_x, None)[0]
            else:
                outputs = model(batch_x, None)

            # normalize batch y
            batch_y = batch_y.reshape(len(batch_y), args.d_feat, -1)
            batch_y = batch_y.permute(0, 2, 1)
            means = batch_y.mean(1, keepdim=True).detach()
            batch_y = batch_y - means
            stdev = torch.sqrt(torch.var(batch_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
            batch_y /= stdev
            f_dim = 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = mse_loss(outputs, batch_y)
            # preds.append(pd.DataFrame({'score': outputs.cpu().numpy(), 'label': batch_y.cpu().numpy(), }, index=index))

        losses.append(loss.item())

    # scores = ic + ndcg[100] + precision[100]
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)

    return np.mean(losses), np.std(losses)


def inference(model, data_loader):

    model.eval()
    mse_loss = torch.nn.MSELoss()
    losses = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        batch_x, batch_y, index, _ = data_loader.get(slc)
        with torch.no_grad():
            if args.output_attention:
                outputs = model(batch_x, None)[0]
            else:
                outputs = model(batch_x, None)

            # normalize batch y
            batch_y = batch_y.reshape(len(batch_y), args.d_feat, -1)
            batch_y = batch_y.permute(0, 2, 1)
            means = batch_y.mean(1, keepdim=True).detach()
            batch_y = batch_y - means
            stdev = torch.sqrt(torch.var(batch_y, dim=1, keepdim=True, unbiased=False) + 1e-5)
            batch_y /= stdev
            f_dim = 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = mse_loss(outputs, batch_y)
        losses.append(loss.item())

    return np.mean(losses), np.std(losses)


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
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_ltsf_loader(args, device=device)

    for times in range(args.repeat):
        pprint('create model...')
        model = get_model(args.model_name)(args)
        
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
            train_epoch(model, optimizer, train_loader, args)
            # save model  after every epoch
            # -------------------------------------------------------------------------
            # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
            # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

            # set smooth step to 1 to avoid smooth operation
            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            # when evaluating, use the avg_params in the current time to evaluate
            model.load_state_dict(avg_params)

            pprint('evaluating...')
            # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
            train_loss, train_loss_std = test_epoch(epoch, model, train_loader, writer, prefix='Train')
            val_loss, val_loss_std = test_epoch(epoch, model, valid_loader, writer, prefix='Valid')
            test_loss, test_loss_std = test_epoch(epoch, model, test_loader, writer, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            pprint('train_loss_std %.6f, valid_loss_std %.6f, test_loss_std %.6f' % (train_loss_std,
                                                                                     val_loss_std, test_loss_std))
            # score equals to ic here
            # pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))

            # load back the current parameters
            model.load_state_dict(params_ckpt)
            val_score = -val_loss

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
        print('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            # do prediction on train, valid and test data
            loss, loss_std = inference(model, eval(name + '_loader'))
            # save the pkl every repeat time
            # pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            pprint(('%s: MSE %.6f') % (name, loss))
            res[name + '-MSE'] = loss

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/' + key: value
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
    parser.add_argument('--d_feat', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--loss_type', default='')
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
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task setup')
    parser.add_argument('--pred_len', default=10, help='should be equal to output_seq_len by default')
    parser.add_argument('--de_norm', default=False, help='add de-normalized in the end of model or not.')

    # training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=10)

    # data
    parser.add_argument('--target', type=str, default='t+0')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='PatchTST')

    # input for dataloader
    parser.add_argument('--data_folder', default='../../csv_data')
    parser.add_argument('--input_seq_len', default=60)
    parser.add_argument('--output_seq_len', default=10)
    parser.add_argument('--outdir', default='../output/ltf/PatchTST_4_layer')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    for prediction, maybe repeat 5, early_stop 10 is enough
    """
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
