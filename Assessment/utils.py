from Explanation.HKUSTsrc import NRSR
from Model.model_pool.models.model import MLP, LSTM, GRU, GAT, ALSTM, KEnhance, relation_GATs


def get_model(model_name):
    a = model_name.upper()
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM
        # return LSTMModel

    if model_name.upper() == 'GRU':
        return GRU
        # return GRUModel
    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == "NRSR":
        return NRSR

    if model_name == "relation_GATs":
        return relation_GATs

    if model_name == "KEnhance":
        return KEnhance