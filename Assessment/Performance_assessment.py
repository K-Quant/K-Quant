from Rating.risk_return_assessment import *

if __name__ == '__main__':
    # input 2
    score_dic = {'return': 5, 'risk': 5, 'max_drawdown': 5, 'Comparison1': [6, 6, 3], 'Comparison2': 70}
    # input3 & input4
    return_preference = 1
    risk_preference = 70
    # input5 - input8
    start_date = '20190601'
    end_date = '20191231'
    start_date_p = '20190601'
    end_date_p = '20191231'
    # input9
    Astock_file = r"/Data/Assessement_data/em_hs_basic_info.json"
    # input10 & input11
    hs300_file = r"/Data/Assessement_data/沪深300指数.json"
    csi500_file = r"/Data/Assessement_data/中证500指数.json"

    # Recommended Stock(s) from the Prediction Model
    select_dic = {'600053.SH': 1}

    C = Get_Score(select_dic, score_dic, return_preference, risk_preference, \
                  start_date, end_date, start_date_p, end_date_p, \
                  Astock_file, hs300_file, csi500_file)
    performance_assessment_results_dict = C.cal_performance_assessment()

    print(performance_assessment_results_dict)
