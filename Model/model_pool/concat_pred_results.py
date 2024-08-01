import os
import pandas as pd

model_list = ['RSR', 'KEnhance', 'LSTM', 'GRU', 'GATs', 'MLP', 'ALSTM', 'SFM', 'HIST']

preds = []
models = os.listdir('output')
res = dict()
for model in model_list:
    
    pred = pd.read_csv('output/' + model + '/pred.csv')
    pred.index = pred['datetime'] + '/' + pred['instrument']
    label = pred['label']
    pred = pred[['score']]
    pred.columns = [f'{model}_score']

    preds.append(pred)

pred = pd.concat(preds + [label], axis=1).sort_index()

pred_old = pd.read_csv('./exp/pred_output/preds.csv', index_col=0)

res = pd.concat((
    pred_old.loc[pred_old.index < pred.index.min()], 
    pred), axis=0)

res.to_csv('./exp/pred_output/preds.csv')