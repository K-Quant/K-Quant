if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#wget https://github.com/chenditc/investment_data/releases/download/2024-04-02/qlib_bin.tar.gz
#tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
#rm qlib_bin.tar.gz

for model_name in RSR_hidy_is HIST GRU LSTM ALSTM SFM
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2024-03-31 \
  --incre_val_start 2024-04-01 \
  --incre_val_end   2024-06-30 \
  --no_test \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2024Q3.log' 2>&1
done
for model_name in GATs MLP
do
python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.0002 --lr_da 0.002 --step 2 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2024-03-31 \
  --incre_val_start 2024-04-01 \
  --incre_val_end   2024-06-30 \
  --no_test \
  --skip_valid_epoch 1 >> logs/'DoubleAdapt_'$model_name'_r2_lr0.0002_lrda0.002_2024Q3.log' 2>&1
done
for model_name in RSR_hidy_is HIST GRU LSTM ALSTM SFM
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2024-06-30 \
  --incre_val_start 2024-07-01 \
  --incre_val_end   2024-09-30 \
  --no_test \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2024Q4.log' 2>&1
done
for model_name in KEnhance GATs MLP
do
python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.0002 --lr_da 0.002 --step 2 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2024-06-30 \
  --incre_val_start 2024-07-01 \
  --incre_val_end   2024-09-30 \
  --no_test \
  --skip_valid_epoch 1 >> logs/'DoubleAdapt_'$model_name'_r2_lr0.0002_lrda0.002_2024Q4.log' 2>&1
done