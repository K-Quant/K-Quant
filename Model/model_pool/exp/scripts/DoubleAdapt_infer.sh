if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

test_end=$1

for model_name in RSR_hidy_is HIST GRU LSTM ALSTM SFM
do
  python -u learn_incre_DoubleAdapt.py --model_name $model_name \
  --lr 0.001 --lr_da 0.01 --step 5 --online_lr "{'lr_da': 0.001}" \
  --test_end   $test_end --reload \
  --skip_valid_epoch 10 >> logs/'DoubleAdapt_'$model_name'_'$test_end'.log' 2>&1
done

model_name=KEnhance
lr_da=0.005
lr_model=0.001
lr_ma=0.0001
python -u learn_incre_DoubleAdapt.py --model_name $model_name \
--lr 0.0002 --lr_da 0.002 --online_lr "{'lr_da': $lr_da, 'lr_model': $lr_model, 'lr_ma': $lr_ma}" \
--test_end        $test_end \
--early_stop 10 --step 2 --reload \
--skip_valid_epoch 1 >> logs/'DoubleAdapt_'$model_name'_r2_lr0.0002_lrda0.002_online_da'$lr_da'_model'$lr_model'_ma'$lr_ma'_'$test_end'.log' 2>&1

model_name=GATs
lr_da=0.001
lr_model=0.001
lr_ma=0.0001
python -u learn_incre_DoubleAdapt.py --model_name $model_name \
--lr 0.0002 --lr_da 0.002 --online_lr "{'lr_da': $lr_da, 'lr_model': $lr_model, 'lr_ma': $lr_ma}" \
--test_end        $test_end \
--early_stop 10 --step 2 --reload \
--skip_valid_epoch 1 >> logs/'DoubleAdapt_'$model_name'_r2_lr0.0002_lrda0.002_online_da'$lr_da'_model'$lr_model'_ma'$lr_ma'_'$test_end'.log' 2>&1

model_name=MLP
lr_da=0.0005
lr_model=0.001
lr_ma=0.001
python -u learn_incre_DoubleAdapt.py --model_name $model_name \
--lr 0.0002 --lr_da 0.002 --online_lr "{'lr_da': $lr_da, 'lr_model': $lr_model, 'lr_ma': $lr_ma}" \
--test_end        $test_end \
--early_stop 10 --step 2 --reload \
--skip_valid_epoch 1 >> logs/'DoubleAdapt_'$model_name'_r2_lr0.0002_lrda0.002_online_da'$lr_da'_model'$lr_model'_ma'$lr_ma'_'$test_end'.log' 2>&1