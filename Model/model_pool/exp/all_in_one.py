import os.path

import pandas as pd

pred_path = 'pred_output'
prefix = 'DoubleAdapt_'
model_pool = 'KEnhance GATs RSR_hidy_is HIST GRU LSTM ALSTM SFM MLP'.split(' ')
years = ['2023', '2024']
all_in_one = []
last_Q = None
for year in years:
    for Q in range(1, 5):
        dfs = None
        for model in model_pool:
            filename = f'{pred_path}/{prefix}{model}_{year}Q{Q}.csv'
            if not os.path.exists(filename):
                dfs = None
                break
            df = pd.read_csv(filename, index_col=[0, 1], header=0)
            if dfs is None:
                dfs = df.rename(columns={'pred': model + '_score'})
            else:
                dfs[model + '_score'] = df['pred']
        if dfs is not None:
            all_in_one.append(dfs)
            print(dfs.head())
            last_Q = f'{year}Q{Q}'
all_in_one = pd.concat(all_in_one)
print(last_Q)
print(all_in_one.tail())
all_in_one.to_csv(f'pred_output/all_in_one_DoubleAdapt.csv')

