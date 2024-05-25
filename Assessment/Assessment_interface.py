from Assessment.Credibility_assessment import run_credibility_assessment
from Assessment.Performance_assessment import run_performance_assessment
from Assessment.run_assessment import prepare_data_and_model, test_get_stocks_recommendation
from Assessment.utils import parse_args, normalize_assessment_results_list

if __name__ == '__main__':
    args = parse_args()

    # model_list = ['LSTM', 'GRU', 'MLP', 'NRSR', 'relation_GATs']
    # # model_list = ['NRSR']
    # explanation_model = "inputGradientExplainer"
    # seq_len_list = [30, 60]
    #
    args.start_date = '2024-03-01'
    args.end_date = '2024-03-10'
    #
    # args.seq_len = 60
    # c_a_r_list = []
    # for seq_len in seq_len_list:
    #     for model in model_list:
    #         h_p_dict = {
    #             "prediction_model": model,
    #             "explanation_model": explanation_model,
    #             "start_date": args.start_date,
    #             "end_date": args.end_date,
    #             "seq_len": seq_len
    #         }
    #         args.model_name = model
    #         args.seq_len = seq_len
    #
    #         data_loader, param_dict, model = prepare_data_and_model(args)
    #         credibility_assessment_results_dict = run_credibility_assessment(param_dict, data_loader, model,
    #                                                                          explanation_model)
    #
    #         c_a_r_list.append((h_p_dict, credibility_assessment_results_dict))
    #
    # n_c_a_r_list = normalize_assessment_results_list(c_a_r_list, num_selection = 5)
    #
    # print(n_c_a_r_list)
    #
    # args.model_name = "NRSR"
    # explanation_model = "inputGradientExplainer"
    # args.seq_len = 60
    # args.num_recommendation_stocks = 3
    # data_loader, param_dict, model = prepare_data_and_model(args)
    # recommend_stocks_list = test_get_stocks_recommendation(param_dict, data_loader, model,
    #                                                        top_n=args.num_recommendation_stocks)  # 输出的是推荐的股票
    #
    # print(recommend_stocks_list)

    select_dict_list = [
        {
            '600061.SH': 0.1,
            '601009.SH': 0.2,
            '601066.SH': 0.1,
            '600519.SH': 0.3,
            '600606.SH': 0.3
        },
        {
            '600061.SH': 0.2,
            '601009.SH': 0.2,
            '600887.SH': 0.4,
            '600132.SH': 0.2,
        },
        {
            '600010.SH': 0.8,
            '600132.SH': 0.1,
            '600489.SH': 0.1

        },
        {
            '600760.SH': 0.3,
            '600000.SH': 0.2,
            '600600.SH': 0.2,
            '601088.SH': 0.3

        },
        {
            '600837.SH': 0.7,
            '601009.SH': 0.2,
            '601066.SH': 0.1,

        },
        {
            '601009.SH': 0.1,
            '601066.SH': 0.5,
            '600132.SH': 0.4
        },
    ]

    # 下面开始计算性能评价
    #更新时间表述
    args.start_date = args.start_date.replace('-', '')
    args.end_date = args.end_date.replace('-', '')
    args.return_preference = 2 # 输入回报偏好
    args.risk_preference = 60 # 输入风险偏好
    p_a_r_list = []
    for select_dict in select_dict_list:
        h_p_dict = {
                        "select_dict": select_dict,
                        "return_preference": args.return_preference,
                        "seq_len": args.risk_preference,
                        "start_date": args.start_date,
                        "end_date": args.end_date,

                    }
        performance_assessment_results_dict = run_performance_assessment(args, select_dict) # 输出性能得分
        print(performance_assessment_results_dict)
        p_a_r_list.append((h_p_dict, performance_assessment_results_dict))

    n_p_a_r_list = normalize_assessment_results_list(p_a_r_list, num_selection=5)
    print(n_p_a_r_list)


