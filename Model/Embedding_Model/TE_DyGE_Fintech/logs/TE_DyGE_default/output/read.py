import numpy as np
import glob
import pandas as pd
import json

npz_data = np.load('/export/data/liane/HKUST_KAISA/mission21_seld/TE_DyGE_fin/logs/TE_DyGE_default/output/default_embs_FinKG_37.npz')
print(npz_data['data'][0],len(npz_data['data']))