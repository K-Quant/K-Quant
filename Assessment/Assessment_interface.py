from Assessment.Performance_assessment import run_performance_assessment
from Assessment.run_assessment import prepare_data_and_model, test_get_stocks_recommendation
from Assessment.utils import parse_args

if __name__ == '__main__':
    args = parse_args()

    args.start_date = '2019-06-01'
    args.end_date = '2019-12-31'
    args.model_name = "NRSR"
    args.seq_len = 60
    args.num_recommendation_stocks = 1

    data_loader, param_dict, model = prepare_data_and_model(args)

    recommend_stocks_list = test_get_stocks_recommendation(param_dict, data_loader, model,
                                                           top_n=args.num_recommendation_stocks) # 输出的是推荐的股票

    print(recommend_stocks_list)

    # 下面开始计算性能评价
    #更新时间表述
    args.start_date = args.start_date.replace('-', '')
    args.end_date = args.end_date.replace('-', '')
    args.return_preference = 1 # 输入回报偏好
    args.risk_preference = 70 # 输入风险偏好
    select_dic = { # 用户自定义输入购入量
        '600053.SH': 1
    }
    performance_assessment_results_dict = run_performance_assessment(args, select_dic) # 输出性能得分

    print(performance_assessment_results_dict)



