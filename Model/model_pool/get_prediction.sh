#!/bin/bash

start_date=$1
end_date=$2
device=$3

models=("RSR" "GATs" "HIST" "GRU" "LSTM" "ALSTM" "SFM" "MLP" "KEnhance")
for model in "${models[@]}"; do
    python exp/learn.py --model_name $model --outdir ./output/$model --device $device --n_epochs 0 --repeat 1 --train_start_date $start_date --train_end_date $end_date --valid_start_date $start_date --valid_end_date $end_date --test_start_date $start_date --test_end_date $end_date --weight_path ./exp/pred_output/models/$model.bin
done

python concat_pred_results.py