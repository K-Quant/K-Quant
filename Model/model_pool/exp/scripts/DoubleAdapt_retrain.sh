if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

wget https://github.com/chenditc/investment_data/releases/download/2024-03-29/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
rm qlib_bin.tar.gz

for model_name in GATs RSR_hidy_is KEnhance HIST GRU LSTM ALSTM SFM MLP
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2023-12-31 \
  --incre_val_start 2024-01-01 \
  --incre_val_end   2024-03-31 \
  --no_test \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2024Q2.log' 2>&1
done