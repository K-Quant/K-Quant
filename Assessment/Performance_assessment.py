from Rating.risk_return_assessment import *

Astock_file = r"./Data/Assessement_data/em_hs_basic_info.json"
# input10 & input11
hs300_file = r"./Data/Assessement_data/沪深300指数.json"
csi500_file = r"./Data/Assessement_data/中证500指数.json"
score_dic = {'return': 5, 'risk': 5, 'max_drawdown': 5, 'Comparison1': [6, 6, 3], 'Comparison2': 70}


def run_performance_assessment(args, select_dic):
    C = Get_Score(select_dic, score_dic, args.return_preference, args.risk_preference, args.start_date, args.end_date, args.start_date, args.end_date,
                  Astock_file, hs300_file, csi500_file)
    performance_assessment_results_dict = C.cal_performance_assessment()

    return performance_assessment_results_dict


