if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#wget https://github.com/chenditc/investment_data/releases/download/2024-03-29/qlib_bin.tar.gz
#tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/crowd_data --strip-components=2
#rm qlib_bin.tar.gz

for model_name in GATs RSR_hidy_is KEnhance HIST GRU LSTM ALSTM SFM MLP
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2022-12-31 \
  --incre_val_start 2023-01-01 \
  --incre_val_end   2023-03-31 \
  --test_start      2023-04-01 \
  --test_end        2023-06-30 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2023Q2.log' 2>&1

  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2023-03-31 \
  --incre_val_start 2023-04-01 \
  --incre_val_end   2023-06-30 \
  --test_start      2023-07-01 \
  --test_end        2023-09-30 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2023Q3.log' 2>&1

  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2023-06-30 \
  --incre_val_start 2023-07-01 \
  --incre_val_end   2023-09-30 \
  --test_start      2023-10-01 \
  --test_end        2023-12-31 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2023Q4.log' 2>&1

  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_end 2023-09-30 \
  --incre_val_start 2023-10-01 \
  --incre_val_end   2023-12-31 \
  --test_start      2024-01-01 \
  --test_end        2024-03-31 \
  --skip_valid_epoch 10 > logs/'DoubleAdapt_'$model_name'_2024Q1.log' 2>&1
done
