from Explanation.HKUSTsrc import NRSR
from Model.model_pool.models.model import MLP, LSTM, GRU, GAT


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

    if model_name.upper() == "NRSR":
        return NRSR