import argparse
from run_assessment import main


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='RSR')
    parser.add_argument('--d_feat', type=int, default=6)
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
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--pred_len', type=int, default=-1, help='the length of pred squence, in regression set to -1')
    parser.add_argument('--de_norm', default=True, help='de normalize or not')

    # training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=5)

    # data
    parser.add_argument('--data_set', type=str, default='csi360')
    parser.add_argument('--target', type=str, default='t+0')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0)
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--start_date', default='2019-01-01')
    parser.add_argument('--end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--name', type=str, default='RSR')

    # input for csi 300
    parser.add_argument('--data_root', default='D:\ProjectCodes\K-Quant\Data')
    parser.add_argument('--market_value_path', default= 'D:\ProjectCodes\K-Quant\Data\csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='D:\ProjectCodes\K-Quant\Data\csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='D:\ProjectCodes\K-Quant\Data\csi300_multi_stock2stock_all.npy')
    parser.add_argument('--stock_index', default='D:\ProjectCodes\K-Quant\Data\csi300_stock_index.npy')
    parser.add_argument('--outdir', default='./output/RSR_all')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)