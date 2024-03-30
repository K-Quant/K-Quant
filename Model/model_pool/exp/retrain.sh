if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

wget https://github.com/chenditc/investment_data/releases/download/2024-03-29/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
rm qlib_bin.tar.gz

year=2023
for Q in 2 3 4
do
for model_name in GATs RSR_hidy_is KEnhance HIST GRU LSTM ALSTM SFM MLP
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --year 2024 --Q 1 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_'$year'Q'$Q'_r5.log' 2>&1
done
done
year=2024
Q=1
for model_name in GATs RSR_hidy_is KEnhance HIST GRU LSTM ALSTM SFM MLP
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --year 2024 --Q 1 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_'$year'Q'$Q'_r5.log' 2>&1
done