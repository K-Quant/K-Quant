mkdir output
mkdir pred_output
python exp/learn.py --model_name MLP --out_dir output/MLP
python exp/learn.py --model_name SFM --out_dir output/SFM
python exp/learn.py --model_name LSTM --out_dir output/LSTM
python exp/learn.py --model_name GRU --out_dir output/GRU
python exp/learn.py --model_name ALSTM --out_dir output/ALSTM
python exp/learn.py --model_name GATs --out_dir output/GATs
python exp/learn.py --model_name RSR --out_dir output/RSR
python exp/learn.py --model_name HIST --out_dir output/HIST --stock_index './data/csi300_stock_index.npy'
python exp/learn.py --model_name KEnhance --out_dir output/KEnhance
