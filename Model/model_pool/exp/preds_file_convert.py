import pandas as pd
# these two files are released from github page
doubleadapt = pd.read_csv('pred_output/all_in_one_DoubleAdapt.csv')
vanilla = pd.read_csv('pred_output/preds.csv')

vanilla[['datetime', 'instrument']] = vanilla['Unnamed: 0'].str.split('/', expand=True)
vanilla = vanilla.drop(columns=['Unnamed: 0'])
vanilla['datetime'] = pd.to_datetime(vanilla['datetime'])
vanilla.set_index(['datetime','instrument'],inplace=True)
vanilla.to_pickle('pred_output/vanilla_preds_latest.pkl')
doubleadapt['datetime']=pd.to_datetime(doubleadapt['datetime'])
doubleadapt.set_index(['datetime', 'instrument'], inplace=True)
doubleadapt.to_pickle('pred_output/da_preds_latest.pkl')