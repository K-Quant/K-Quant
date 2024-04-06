import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Assessment.Credibility_assessment import run_credibility_assessment
from Assessment.Performance_assessment import run_performance_assessment
from Assessment.run_assessment import prepare_data_and_model, test_get_stocks_recommendation
from Assessment.utils import parse_args, normalize_assessment_results_list

if __name__ == '__main__':
    args = parse_args()

    model_list = ['LSTM', 'GRU', 'MLP', "NRSR", 'relation_GATs']
    explanation_model = "inputGradientExplainer"
    seq_len_list = [30,60]

    args.start_date = '2019-06-01'
    args.end_date = '2019-06-15'

    args.seq_len = 60
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
    
    #         data_loader, param_dict, model = prepare_data_and_model(args)
    #         credibility_assessment_results_dict = run_credibility_assessment(param_dict, data_loader, model,
    #                                                                          explanation_model)
    
    #         c_a_r_list.append((h_p_dict, credibility_assessment_results_dict))
    
    # n_c_a_r_list = normalize_assessment_results_list(c_a_r_list)
    
    # print(n_c_a_r_list)

    args.model_name = "NRSR"
    # explanation_model = "inputGradientExplainer"
    args.seq_len = 30
    c_a_r_list = []
    for seq_len in seq_len_list:
        for model in model_list:
            h_p_dict = {
                "prediction_model": model,
                "explanation_model": explanation_model,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "seq_len": seq_len
            }
            args.model_name = model
            args.seq_len = seq_len

            data_loader, param_dict, model = prepare_data_and_model(args)
            credibility_assessment_results_dict = run_credibility_assessment(param_dict, data_loader, model,
                                                                             explanation_model)

            c_a_r_list.append((h_p_dict, credibility_assessment_results_dict))

    n_c_a_r_list = normalize_assessment_results_list(c_a_r_list, num_selection = 5)

    print(n_c_a_r_list)

    args.model_name = "NRSR"
    explanation_model = "inputGradientExplainer"
    args.seq_len = 60
    args.num_recommendation_stocks = 3
    data_loader, param_dict, model = prepare_data_and_model(args)
    recommend_stocks_list = test_get_stocks_recommendation(param_dict, data_loader, model,
                                                           top_n=args.num_recommendation_stocks)  # 输出的是推荐的股票
    

    print(recommend_stocks_list)

    # select_dict_list = [
    #     {
    #         '002460.SZ': 3,

    #     },
    #     {
    #         '600009.SH': 1,

    #     },
    #     {
    #         '600000.SH': 100,

    #     },
    #     {
    #         '600015.SH': 1,
    #     },
    #     {
    #         '600703.SH': 1,
    #     },
    #     {
    #         '300072.SZ': 1
    #     },
    # ]

    # # 下面开始计算性能评价
    # #更新时间表述
    # args.start_date = args.start_date.replace('-', '')
    # args.end_date = args.end_date.replace('-', '')
    # args.return_preference = 0 # 输入回报偏好
    # args.risk_preference = 90 # 输入风险偏好
    # p_a_r_list = []
    # for select_dict in select_dict_list:
    #     h_p_dict = {
    #                     "select_dict": select_dict,
    #                     "return_preference": args.return_preference,
    #                     "seq_len": args.risk_preference,
    #                     "start_date": args.start_date,
    #                     "end_date": args.end_date,

    #                 }
    #     performance_assessment_results_dict = run_performance_assessment(args, select_dict) # 输出性能得分
    #     print(performance_assessment_results_dict)
    #     p_a_r_list.append((h_p_dict, performance_assessment_results_dict))

    # n_p_a_r_list = normalize_assessment_results_list(p_a_r_list)
    # print(n_p_a_r_list)
    n_p_a_r_list = normalize_assessment_results_list(p_a_r_list, num_selection=5)
    print(n_p_a_r_list)


