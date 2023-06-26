import pandas as pd
import glob
import os
import argparse
from tqdm import tqdm


def create_ltsf_loader(args):
    """
    in create ltsf loader, we need to build a dataset that suits for long-term time series forecasting
    :param args:
        args.train_start_date
        args.train_end_date
        args.valid_start_date
        args.valid_end_date
        args.test_start_date
        args.test_end_date
        args.input_seq_len
        args.output_seq_len
        args.data_folder
        args.stock_dict
    :return:
        dataloaders, that every batch contains the time series of different stocks in the same day and the
    pred is also time series
    """
    file_dict = glob.glob(os.path.join(args.data_folder, "*.csv"))
    for file in file_dict:
        x = pd.read_csv(file).set_index('date')

        break

    return None


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_folder', default='../../csv_data')
        args = parser.parse_args()
        return args

    args = parse_args()
    create_ltsf_loader(args)