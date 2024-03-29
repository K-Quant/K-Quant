if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

for model_name in GATs RSR_hidy_is KEnhance HIST GRU LSTM MLP ALSTM SFM
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_start 2008-01-01 --incre_train_end 2020-01-01 \
  --incre_val_start 2020-01-01 --incre_val_end 2022-12-31 \
  --test_start 2021-01-01 --test_end 2023-06-30 \
  --skip_valid_epoch 5 > logs/'DoubleAdapt_'$model_name'_2023-06-30_r5.log' 2>&1
done
for model_name in GATs RSR_hidy_is KEnhance HIST
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --incre_train_start 2008-01-01 --incre_train_end 2020-01-01 \
  --incre_val_start 2020-01-01 --incre_val_end 2022-12-31 \
  --test_start 2021-01-01 --test_end 2023-06-30 \
  --skip_valid_epoch 5 --naive True > logs/'IL_'$model_name'_2023-06-30_r5.log' 2>&1
done