<<BLOCK
  now we have basic training and incremental training cn_data
  we need to do:
  1. update local qlib cn_data
  2. donwlad and process ready-to-use preds files
  3. do ensemble based on preds files
  4. do evaluation for different time scale and industries
  5. do backtest for different time scale and industries
  finally, please delete unnecessary files to keep the branch clean.

  ATTENTION: Please be careful for the time stamp and file name. Some of the time stamp arguments in python scripts need
  to be modified for latest result.
BLOCK

local=$1
end_date=$2

start_date="2024-07-09"
device="cuda:0"

# step 1
response=$(curl -s https://api.github.com/repos/chenditc/investment_data/releases/latest)
version=$(echo "$response" | grep 'tag_name' | cut -d'"' -f4)
wget https://github.com/chenditc/investment_data/releases/download/$version/qlib_bin.tar.gz
# the target folder need to be created first
tar -zxvf qlib_bin.tar.gz -C ../../../stock_model/qlib_data/cn_data --strip-components=1
rm qlib_bin.tar.gz

# step 2 make sure you have download latest pred file and put them into pred_output folder
# if not, here is the url: https://github.com/Hexagram-Sun/stock_preds/releases/tag/preds

response=$(curl -s https://api.github.com/repos/Hexagram-Sun/stock_preds/releases/latest)
url=$(echo "$response" | grep 'body' | cut -d'"' -f4 | cut -d'(' -f2 | cut -d')' -f1)
wget $url
tar -zxvf models_and_preds.tar.gz -C exp/pred_output --strip-components=1

if [ $local -gt 0 ]; then
  bash get_prediction.sh $start_date $end_date $device
  cd exp/
  sh scripts/DoubleAdapt_infer.sh $end_date
  python all_in_one.py
  cd ../
fi
cd exp/
python preds_file_convert.py
cd ../

# step 3 ensemble learning
python exp/ensemble_inference.py --reference_file 'pred_output/vanilla_preds_latest.pkl' \
--saved_file 'pred_output/ensemble_preds_latest.pkl'

# step 4 evaluation for each industry in different time scope
mkdir pred_output/platform_data

for industry in 'all' 'dianzi' 'yiyaoshengwu' 'yinhang' 'feiyinjinrong' 'dianlishebei' 'jisuanji'
do
  # regular training evaluation
  python exp/evaluations.py --evaluation_start_date '2023-04-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --report_file "pred_output/platform_data/evaluation_$industry-12.pkl"
  python exp/evaluations.py --evaluation_start_date '2023-11-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --report_file "pred_output/platform_data/evaluation_$industry-6.pkl"
  python exp/evaluations.py --evaluation_start_date '2024-02-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --report_file "pred_output/platform_data/evaluation_$industry-3.pkl"

  # double adapt evaluation
  python exp/evaluations.py --evaluation_start_date '2023-05-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/da_preds_latest.pkl'\
  --report_file "pred_output/platform_data/da_evaluation_$industry-12.pkl"
  python exp/evaluations.py --evaluation_start_date '2023-11-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/da_preds_latest.pkl'\
  --report_file "pred_output/platform_data/da_evaluation_$industry-6.pkl"
  python exp/evaluations.py --evaluation_start_date '2024-02-01' --evaluation_end_date '2024-04-30'\
  --industry_category $industry --predicted_file 'pred_output/da_preds_latest.pkl'\
  --report_file "pred_output/platform_data/da_evaluation_$industry-3.pkl"

done

# step 5 backtest first for every industry top5 strategy in 3 time scope

for industry in 'dianzi' 'yiyaoshengwu' 'yinhang' 'feiyinjinrong' 'dianlishebei' 'jisuanji'
do
  python exp/backtest.py --prefix "pred_output/platform_data/backtest_data_$industry-" --backtest_start_date '2023-04-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_$industry-12_5.pkl" --time_scope "12_5"\
  --industry_category $industry

  python exp/backtest.py --prefix "pred_output/platform_data/backtest_data_$industry-" --backtest_start_date '2023-11-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_$industry-6_5.pkl" --time_scope "6_5"\
  --industry_category $industry

  python exp/backtest.py --prefix "pred_output/platform_data/backtest_data_$industry-" --backtest_start_date '2024-02-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_$industry-3_5.pkl" --time_scope "3_5"\
  --industry_category $industry

  python exp/backtest.py --prefix "pred_output/platform_data/da_backtest_data_$industry-" --backtest_start_date '2023-04-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_$industry-12_5.pkl" --time_scope "12_5"\
  --industry_category $industry

  python exp/backtest.py --prefix "pred_output/platform_data/da_backtest_data_$industry-" --backtest_start_date '2023-11-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_$industry-6_5.pkl" --time_scope "6_5"\
  --industry_category $industry

  python exp/backtest.py --prefix "pred_output/platform_data/da_backtest_data_$industry-" --backtest_start_date '2024-02-01'\
  --backtest_end_date '2024-04-30' --topk 5 --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_$industry-3_5.pkl" --time_scope "3_5"\
  --industry_category $industry
done

for strategy in 30 50 100
do
  python exp/backtest.py --prefix 'pred_output/platform_data/backtest_data_all-' --backtest_start_date '2023-04-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_all-12_$strategy.pkl" --time_scope "12_$strategy"\
  --industry_category 'all'

  python exp/backtest.py --prefix 'pred_output/platform_data/backtest_data_all-' --backtest_start_date '2023-11-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_all-6_$strategy.pkl" --time_scope "6_$strategy"\
  --industry_category 'all'

  python exp/backtest.py --prefix 'pred_output/platform_data/backtest_data_all-' --backtest_start_date '2024-02-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/ensemble_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/backtest_result_all-3_$strategy.pkl" --time_scope "3_$strategy"\
  --industry_category 'all'

    python exp/backtest.py --prefix 'pred_output/platform_data/da_backtest_data_all-' --backtest_start_date '2023-04-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_all-12_$strategy.pkl" --time_scope "12_$strategy"\
  --industry_category 'all'

  python exp/backtest.py --prefix 'pred_output/platform_data/da_backtest_data_all-' --backtest_start_date '2023-11-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_all-6_$strategy.pkl" --time_scope "6_$strategy"\
  --industry_category 'all'

  python exp/backtest.py --prefix 'pred_output/platform_data/da_backtest_data_all-' --backtest_start_date '2024-02-01'\
  --backtest_end_date '2024-04-30' --topk $strategy --predicted_file 'pred_output/da_preds_latest.pkl'\
  --backtest_file "pred_output/platform_data/da_backtest_result_all-3_$strategy.pkl" --time_scope "3_$strategy"\
  --industry_category 'all'
done
