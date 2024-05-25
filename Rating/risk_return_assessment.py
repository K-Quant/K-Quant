# -*- coding: utf-8 -*-

import math
import pandas as pd
import tushare as ts
from collections import Counter
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
import json
import time
from tqdm import tqdm

token = '3c00afeeeffc6b731cb5fdc3881e19796524d7f30d25842da85512e4'
ts.set_token(token)
pro = ts.pro_api()


class Assessment(object):
    """
    select_dic: 智能投顾模型给出的投资组合。type:dic{stock_code1:weights1, stock_code2:weights2,...}。
    score_dic: 用户自定义的（也可由开发者设定好的）各考察方面的打分权重。type:dic{}。
    return_preference: 用户自定义的收益偏好，例如70%。
    risk_preference: 用户自定义的风险偏好，例如50%。
    start_date: 用户自定义的（也可由开发者设定好的）用于计算的日线数据的开始时间戳。
    end_date: 用户自定义的（也可由开发者设定好的）用于计算的日线数据的结束时间戳。
    start_date_p: （若想应用到预测数据时设置）用户自定义的（也可由开发者设定好的）用于计算的日线数据的开始时间戳。例如'20211201'。
    end_date_p:（若想应用到预测数据时设置）用户自定义的（也可由开发者设定好的）用于计算的日线数据的结束时间戳。
    Astock: A股的“股票代码-行业”列表。
    hs300_list: 沪深300指数中的“股票代码-行业”列表。
    csi500_list: 中证500指数中的“股票代码-行业”列表。
    df_price: 涉及股票的日线数据，例如“close_price”。
    df_pctchg: 涉及股票的日线数据，例如“pct_chg”。
    df_price_base: 常用指数（例如沪深300，中证500）的日线数据，例如“close_price”。
    df_pctchg_base:  常用指数（例如沪深300，中证500）的日线数据，例如“pct_chg”。
    
    """

    def __init__(self, select_dic, score_dic, return_preference, risk_preference, \
                 start_date, end_date, start_date_p, end_date_p, \
                 Astock, hs300_list, csi500_list, \
                 df_price, df_pctchg, df_price_base, df_pctchg_base):

        self.select_dic = select_dic
        self.score_dic = score_dic
        self.return_preference = return_preference
        self.risk_preference = risk_preference
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_p = start_date_p
        self.end_date_p = end_date_p
        self.Astock = Astock
        self.hs300_list = hs300_list
        self.csi500_list = csi500_list
        self.df_price = df_price
        self.df_pctchg = df_pctchg

    def get_annualized_return(self, df):
        """
        计算年化收益率。type: dict. {'ts_code': a_r}.
        """
        annualized_return = df.apply(
            lambda x: round((math.pow(x.tail(1).values[0] / x.head(1).values[0], (252 / df.shape[0])) - 1) * 100, 2))
        a_r = dict(zip(df.columns, annualized_return))
        return a_r

    def get_annualized_volatility(self, df):
        """
        计算年化波动率。type: dict. {'ts_code': a_v}.
        """
        annualized_volatility = df.apply(lambda x: round(x.std() * math.sqrt(252), 2))
        a_v = dict(zip(df.columns, annualized_volatility))
        return a_v

    def get_max_drawdown_static(self, df):
        """
        计算静态最大回撤率。type: dict. {'ts_code': m_d}.
        """
        max_drawdown = df.apply(lambda x: round((1 - x / x.max()).max() * 100, 2))
        m_d = dict(zip(df.columns, max_drawdown))
        return m_d

    def get_max_drawdown_dyna(self, df):
        """
        计算动态最大回撤率。type: data frame. 
        """
        m_d = pd.DataFrame()
        for i in df.columns:
            df1 = df[i]
            max_dropdown = []
            for j in range(1, len(df1) + 1):
                max_dropdown.append(round((1 - df1[:j] / df1[:j].max()).max() * 100, 2))
            m_d[i] = max_dropdown
        m_d = m_d.set_index(df.index)
        return m_d

    def get_scaled_return(self, df):
        """
        计算累计收益率。type: data frame. 
        """
        scaled_df = df.apply(lambda x: x / x.head(1).values[0])
        return scaled_df

    def get_return_daily(self, df):
        """
        计算每日收益率。type: data frame. 
        """
        df_daily = df.apply(lambda x: x / x.shift(1)).dropna()
        return df_daily

    def get_sharpe_ratio(self, df1, df2, risk_free_return):
        """
        计算夏普比率。type: dict. {'ts_code': s_r}.
        df1: df_price
        df2: df_pctchg
        """
        a_r = df1.apply(lambda x: math.pow(x.tail(1).values[0] / x.head(1).values[0], (252 / df1.shape[0])) - 1)
        a_v = df2.apply(lambda x: x.std())
        sharpe_ratio = round((a_r - risk_free_return) / a_v * 100, 2)
        s_r = dict(zip(df1.columns, sharpe_ratio))
        return s_r

    def get_information_ratio(self, a_r, a_r_base, a_r_daily, a_r__daily_base):
        """
        计算信息比率。type: float.
        a_r: 投资组合年化收益率
        a_r_base: 基准的年化收益率
        a_r_daily: 投资组合的每日收益率
        a_r__daily_base: 基准的每日收益率
        """
        i_r = round((a_r - a_r_base) / ((a_r_daily - a_r__daily_base).std() * 100), 2)
        return i_r

    def percentage(self):
        """
        计算所选股票组合占各指数的比例。
        """
        hs300_p = round(len(list(set(self.select_dic.keys()) & set(self.hs300_list))) / len(self.select_dic), 1)
        csi500_p = round(len(list(set(self.select_dic.keys()) & set(self.csi500_list))) / len(self.select_dic), 1)
        rest_p = round(1 - (hs300_p + csi500_p), 1)
        perc = {'沪深300': hs300_p,
                '中证500': csi500_p,
                '剩余股票': rest_p}
        return perc

    def percentage_industry(self):
        """
        计算投资组合中，除了被沪深300指数、中证500指数包含的，剩余股票占各行业的比例。
        """
        rest_stock = list(
            set(self.select_dic.keys()).difference(set(self.hs300_list)).difference(set(self.csi500_list)))
        prop = Counter([self.Astock[self.Astock['ts_code'] == code]['industry'].values[0] for code in rest_stock])
        for key in prop.keys():
            prop[key] /= len(rest_stock)
            prop[key] = round(prop[key], 3)
        return prop

    def values(self, prop):
        """
        用投资组合中所有股票的日线数据计算累积收益率和动态最大相对回撤。
        """
        a_r = self.get_scaled_return(self.df_price)
        m_d = self.get_max_drawdown_dyna(self.df_price)

        select_price = 0
        select_max = 0
        for code in self.select_dic.keys():
            if code in a_r:
                select_price += a_r[code] * self.select_dic[code]
                select_max += m_d[code] * self.select_dic[code]

        # 用投资组合中剩余股票所在行业的日线数据计算部分累积收益率和动态最大相对回撤。
        rest_price = 0
        rest_max = 0
        for i in list(set(self.select_dic.keys()).difference(set(self.hs300_list)).difference(set(self.csi500_list))):
            # 某行业
            industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
            # 某行业所有股票名
            stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values

            # 某行业的所有股票日线数据
            df_return = pd.DataFrame()
            df_max_d = pd.DataFrame()
            for code in set(stock_list) & set(a_r.keys()):
                df_return[code] = a_r[code]
                df_max_d[code] = m_d[code]
            rest_price += df_return.mean(axis=1) * prop[industry]
            rest_max += df_max_d.mean(axis=1) * prop[industry]
        return select_price, select_max, rest_price, rest_max

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def bonus(self, x):
        """
        根据现有的同行业股票打分，赋予bonus。
        """
        return 5 + 5 * self.sigmoid(x - self.score_dic['Comparison2'] * 0.01) + 5 * self.sigmoid(
            x - self.score_dic['Comparison2'] * 0.05) \
            + 5 * self.sigmoid(x - self.score_dic['Comparison2'] * 0.09) + 10 * self.sigmoid(
                x - self.score_dic['Comparison2'] * 0.15) \
            + 10 * self.sigmoid(x - self.score_dic['Comparison2'] * 0.30) + 10 * self.sigmoid(
                x - self.score_dic['Comparison2'] * 0.55) \
            + 10 * self.sigmoid(x - self.score_dic['Comparison2'] * 0.85)

    def cal_performance_assessment(self, df_price_base, df_pctchg_base):

        performance_assessment_results_dict = {}
        # 预测集 （根据日期划分数据集）
        df_price_pred = self.df_price[self.df_price.index >= self.start_date_p]
        df_pctchg_pred = self.df_pctchg[self.df_pctchg.index >= self.start_date_p]
        df_price_base_pred = df_price_base[df_price_base.index >= self.start_date_p]
        df_pctchg_base_pred = df_pctchg_base[df_pctchg_base.index >= self.start_date_p]

        # ---------------------- 用户偏好 (可用历史数据做，也可用预测数据做) ----------------------#
        a_r = self.get_annualized_return(df_price_pred)
        a_v = self.get_annualized_volatility(df_pctchg_pred)
        m_d = self.get_max_drawdown_static(df_price_pred)
        record_ar = 0
        record_av = 0
        record_md = 0
        for i in self.select_dic.keys():
            if i in a_r:
                # 年化收益率
                record_ar += a_r[i] * self.select_dic[i]
                # 年化波动率
                record_av += a_v[i] * self.select_dic[i]
                # （静态的）最大回撤率
                record_md += m_d[i] * self.select_dic[i]
        a_r_base = self.get_annualized_return(df_price_base_pred)

        # 计算用户的收益偏好得分
        if record_ar >= self.return_preference:
            score_p11 = self.score_dic['return'] + math.log(record_ar)
        else:
            score_p11 = 0

        # 计算用户风险偏好得分
        if record_av <= self.risk_preference:
            score_p12 = self.score_dic['risk'] + math.log(self.risk_preference - record_av + 1)
        else:
            score_p12 = 0
        # 计算用户投资体验感得分
        if record_md <=  self.risk_preference:
            score_p13 = self.score_dic['max_drawdown']
        else:
            score_p13 = 0

        performance_assessment_results_dict["用户收益偏好得分"] = score_p11
        performance_assessment_results_dict["用户风险偏好得分"] = score_p12
        performance_assessment_results_dict["用户投资体验感得分"] = score_p13

        # ---------------------- 市场表现：业绩基准 v.s. 基本指数 (可用历史数据，也可用预测数据) ----------------------#
        # 查看投资组合所含股票占各指数比例。
        perc = self.percentage()
        # 查看剩余股票占各行业比例。
        prop = self.percentage_industry()

        # 基本指数
        base_scaled_return = self.get_scaled_return(df_price_base)
        base_max_drawdown = self.get_max_drawdown_dyna(df_price_base)

        # 投资组合 & 业绩基准
        select_price, select_max, rest_price, rest_max = self.values(prop)
        base_price = base_scaled_return['hs300'] * perc['沪深300'] + base_scaled_return['csi500'] * perc[
            '中证500'] + rest_price * perc['剩余股票']
        base_max = base_max_drawdown['hs300'] * perc['沪深300'] + base_max_drawdown['csi500'] * perc[
            '中证500'] + rest_max * perc['剩余股票']

        df = pd.DataFrame({'投资组合': select_price,
                           '业绩基准': base_price,
                           '投资组合/业绩基准': select_price / base_price,
                           '相对最大回撤': select_max / base_max,
                           '沪深300': base_scaled_return['hs300'],
                           '中证500': base_scaled_return['csi500']})
        df['相对最大回撤'][0] = 0

        # 相关性
        corr = df.corr(method='pearson')

        # 夏普比率 （可由专业人士自定义为其他数值）
        risk_free_return = 0.04

        # 投资组合的夏普比率
        s_r = self.get_sharpe_ratio(self.df_price, self.df_pctchg, risk_free_return)
        select_sharpe = 0
        for code in self.select_dic.keys():
            if code in s_r:
                select_sharpe += s_r[code] * self.select_dic[code]

        base_s_r = self.get_sharpe_ratio(df_price_base, df_pctchg_base, risk_free_return)

        print('-' * 90)

        if corr['投资组合']['业绩基准'] >= 0.70:
            if select_sharpe > base_s_r['hs300']:
                score_sub11 = self.score_dic['Comparison1'][0] + math.log(select_sharpe - base_s_r['hs300'] + 1)
            else:
                score_sub11 = 0

            if select_sharpe > base_s_r['csi500']:
                score_sub12 = self.score_dic['Comparison1'][1] + math.log(select_sharpe - base_s_r['csi500'] + 1)
            else:
                score_sub12 = 0

            # 业绩基准的夏普比率
            rest_sharpe = 0
            for i in list(
                    set(self.select_dic.keys()).difference(set(self.hs300_list)).difference(set(self.csi500_list))):
                # 某行业
                industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
                # 某行业所有股票名
                stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values
                # 某行业的所有股票日线数据
                sharpe_value = 0
                for code in set(stock_list) & set(s_r.keys()):  # tqdm()
                    sharpe_value += s_r[code]
                rest_sharpe += (sharpe_value / len(set(stock_list) & set(s_r.keys()))) * prop[industry]

            base_sharpe = base_s_r['hs300'] * perc['沪深300'] + base_s_r['csi500'] * perc['中证500'] + \
                          rest_sharpe * perc['剩余股票']

            if select_sharpe > base_sharpe:
                score_sub13 = self.score_dic['Comparison1'][2] + math.log(select_sharpe - base_sharpe + 1)
            else:
                score_sub13 = 0
            score_sub1 = score_sub11 + score_sub12 + score_sub13
        else:

            if select_sharpe > base_s_r['hs300']:
                score_sub11 = sum(self.score_dic['Comparison1']) / 2 + math.log(select_sharpe - base_s_r['hs300'] + 1)
            else:
                score_sub11 = 0

            if select_sharpe > base_s_r['csi500']:
                score_sub12 = sum(self.score_dic['Comparison1']) / 2 + math.log(select_sharpe - base_s_r['csi500'] + 1)
            else:
                score_sub12 = 0
            score_sub13 = score_sub11 + score_sub12
            score_sub1 = score_sub13

        performance_assessment_results_dict["沪深300指数比较得分"] = score_sub11
        performance_assessment_results_dict["中证500指数比较得分"] = score_sub12
        performance_assessment_results_dict["基准比较得分"] = score_sub13

        # ---------------------- 市场表现：同类股票 ----------------------#

        calender = [1, 5, 10,30, 60, 90, 120, 180, 252]  # 可以自定义设置：近半个月、一个月、两个月、三个月、六个月、一年
        # temp = pro.daily(ts_code = [*self.select_dic.keys()][0], start_date = self.start_date, end_date = self.end_date_p)
        # 可以检验的近n天数
        pre_test = [i for i in calender if i <= len(self.df_price)]
        indi_score = self.score_dic['Comparison2'] * (1 / len(self.select_dic)) * (1 / len(pre_test))
        # print('小分值:%0.2f' %indi_score)

        print('-' * 90)
        rank_dic = {}
        for i in self.select_dic.keys():
            if i in self.Astock['ts_code'].values and i in a_r:
                # 所选股的所属行业
                industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
                # 某行业所有股票名
                stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values
                days = []
                for day in pre_test:
                    temp1 = self.get_annualized_return(self.df_price[-day:])
                    temp2 = self.get_annualized_volatility(self.df_pctchg[-day:])
                    # 记录每个股票的年化收益、年化波动
                    record_ar = {}
                    record_av = {}
                    for code in set(stock_list) & set(temp1.keys()):
                        record_ar[code] = temp1[code]
                        record_av[code] = temp2[code]
                    # 所选股在同行业股票中top排名
                    s1 = dict(sorted(record_ar.items(), key=lambda item: item[1], reverse=True))
                    rank_ar = list(s1.keys()).index(i) + 1
                    s2 = dict(sorted(record_av.items(), key=lambda item: item[1]))
                    rank_av = list(s2.keys()).index(i) + 1
                    print('%s: 近%i天年化收益率位于同行业股票Top%i%%(%i/%i), 年化波动率位于同行业股票Top%i%% (%i/%i)。' \
                          % (
                              i, day, rank_ar / len(s1) * 100, rank_ar, len(s1), rank_av / len(s2) * 100, rank_av,
                              len(s2)))

                    if int(rank_ar / len(s1) * 100) <= (100 - self.return_preference) and int(
                            rank_av / len(s2) * 100) <= self.risk_preference:
                        days.append(day)

                rank_dic[i] = days
        # print(indi_score,rank_dic)

        score_sub2 = 0
        for key in rank_dic.keys():
            score_sub2 += indi_score * len(rank_dic[key])

        if score_sub1 > 0 and score_sub2 > 0:
            score_sub2_bonus = self.bonus(score_sub2)
        else:
            score_sub2_bonus = 0

        performance_assessment_results_dict["同行业股票比较对比得分"] = score_sub2_bonus

        return performance_assessment_results_dict

    def full_model(self, df_price_base, df_pctchg_base):

        # 为了减少调用tushare读取数据的次数，这几行注释掉的代码已由直接在函数中输入数据替代。若不想额外下载数据存成csv，可以恢复下面注释代码的运行。

        # 沪深300指数 & 中证500指数
        # hs300 = pro.index_daily(ts_code='399300.SZ', start_date = self.start_date, end_date = self.end_date_p)
        # csi500 = pro.index_daily(ts_code='399905.SZ', start_date = self.start_date, end_date = self.end_date_p)

        # df_price_base = pd.DataFrame({'trade_date': hs300['trade_date'], 'hs300': hs300['close'], 'csi500': csi500['close']})
        # df_pctchg_base = pd.DataFrame({'trade_date': hs300['trade_date'], 'hs300': hs300['pct_chg'], 'csi500': csi500['pct_chg']})

        # df_price_base = df_price_base.sort_values(by="trade_date")
        # df_price_base.set_index('trade_date', inplace=True)
        # df_price_base.index = pd.to_datetime(df_price_base.index, format = '%Y%m%d').strftime('%Y-%m-%d')
        # df_price_base.index = pd.to_datetime(df_price_base.index)
        # df_pctchg_base = df_pctchg_base.sort_values(by="trade_date")
        # df_pctchg_base.set_index('trade_date', inplace=True)
        # df_pctchg_base.index = pd.to_datetime(df_pctchg_base.index, format = '%Y%m%d').strftime('%Y-%m-%d')
        # df_pctchg_base.index = pd.to_datetime(df_pctchg_base.index)

        # 预测集 （根据日期划分数据集）
        df_price_pred = self.df_price[self.df_price.index >= self.start_date_p]
        df_pctchg_pred = self.df_pctchg[self.df_pctchg.index >= self.start_date_p]
        df_price_base_pred = df_price_base[df_price_base.index >= self.start_date_p]
        df_pctchg_base_pred = df_pctchg_base[df_pctchg_base.index >= self.start_date_p]

        # ---------------------- 用户偏好 (可用历史数据做，也可用预测数据做) ----------------------#
        a_r = self.get_annualized_return(df_price_pred)
        a_v = self.get_annualized_volatility(df_pctchg_pred)
        m_d = self.get_max_drawdown_static(df_price_pred)
        record_ar = 0
        record_av = 0
        record_md = 0
        for i in self.select_dic.keys():
            if i in a_r:
                # 年化收益率
                record_ar += a_r[i] * self.select_dic[i]
                # 年化波动率
                record_av += a_v[i] * self.select_dic[i]
                # （静态的）最大回撤率
                record_md += m_d[i] * self.select_dic[i]
        print('-' * 90)

        a_r_base = self.get_annualized_return(df_price_base_pred)
        print('投资策略在投资周期的年化收益率比沪深300指数高%0.2f个点。' % (record_ar - a_r_base['hs300']))
        print('投资策略在投资周期的年化收益率比中证500指数高%0.2f个点。' % (record_ar - a_r_base['csi500']))
        print()

        score_p11 = score_p12 = score_p13 = 0
        if record_ar >= self.return_preference:
            score_p11 = self.score_dic['return'] + math.log(record_ar)
            print('评分：得%0.4f分。投资策略在投资周期的年化收益率%0.2f%%，可以满足用户的收益偏好（%0.2f%%）。' % (
            score_p11, record_ar, self.return_preference))
        else:
            print('评分：未得分。投资策略在投资周期的年化收益率为%0.2f%%，无法满足用户的收益偏好（%0.2f%%）。' % (
            record_ar, self.return_preference))

        if record_av <= self.risk_preference:
            score_p12 = self.score_dic['risk'] + math.log(self.risk_preference - record_av + 1)
            print('评分：得%0.4f分。投资策略在投资周期的年化波动率为%0.2f%%，可以满足用户的风险偏好（%0.2f%%）。' % (
            score_p12, record_av, self.risk_preference))
        else:
            print('评分：未得分。投资策略在投资周期的年化波动率为%0.2f%%，无法满足用户的风险偏好（%0.2f%%）。' % (
            record_av, self.risk_preference))

        if record_md <= self.risk_preference:
            score_p13 = self.score_dic['max_drawdown']
            print(
                '评分：得%0.4f分。投资策略在投资周期的最大回撤率%0.2f%%，可以满足用户的风险偏好（%0.2f%%），用户投资体验感较好。' % (
                score_p13, record_md, self.risk_preference))
        else:
            print(
                '评分：未得分。投资策略在投资周期的最大回撤率为%0.2f%%，无法满足用户的风险偏好（%0.2f%%），用户投资体验感较差。' % (
                record_md, self.risk_preference))
        print()
        score_p1 = score_p11 + score_p12 + score_p13
        print('用户偏好：满分%0.4f，得分（含bonus）%0.4f。' % (
        (self.score_dic['return'] + self.score_dic['risk'] + self.score_dic['max_drawdown']), score_p1))

        # ---------------------- 市场表现：业绩基准 v.s. 基本指数 (可用历史数据，也可用预测数据) ----------------------#
        # 查看投资组合所含股票占各指数比例。
        perc = self.percentage()
        # 查看剩余股票占各行业比例。
        prop = self.percentage_industry()

        # 基本指数
        base_scaled_return = self.get_scaled_return(df_price_base)
        base_max_drawdown = self.get_max_drawdown_dyna(df_price_base)

        # 投资组合 & 业绩基准
        select_price, select_max, rest_price, rest_max = self.values(prop)
        base_price = base_scaled_return['hs300'] * perc['沪深300'] + base_scaled_return['csi500'] * perc[
            '中证500'] + rest_price * perc['剩余股票']
        base_max = base_max_drawdown['hs300'] * perc['沪深300'] + base_max_drawdown['csi500'] * perc[
            '中证500'] + rest_max * perc['剩余股票']

        df = pd.DataFrame({'投资组合': select_price,
                           '业绩基准': base_price,
                           '投资组合/业绩基准': select_price / base_price,
                           '相对最大回撤': select_max / base_max,
                           '沪深300': base_scaled_return['hs300'],
                           '中证500': base_scaled_return['csi500']})
        df['相对最大回撤'][0] = 0

        # 相关性
        corr = df.corr(method='pearson')

        # 夏普比率 （可由专业人士自定义为其他数值）
        risk_free_return = 0.04

        # 投资组合的夏普比率
        s_r = self.get_sharpe_ratio(self.df_price, self.df_pctchg, risk_free_return)
        select_sharpe = 0
        for code in self.select_dic.keys():
            if code in s_r:
                select_sharpe += s_r[code] * self.select_dic[code]

        base_s_r = self.get_sharpe_ratio(df_price_base, df_pctchg_base, risk_free_return)

        print('-' * 90)
        score_sub11 = score_sub12 = score_sub13 = 0
        if corr['投资组合']['业绩基准'] >= 0.70:
            if select_sharpe > base_s_r['hs300']:
                score_sub11 = self.score_dic['Comparison1'][0] + math.log(select_sharpe - base_s_r['hs300'] + 1)
                print('评分：得%0.4f分。投资策略在投资周期的夏普比率为%0.2f%%，可以跑赢沪深300指数（%0.2f%%）。' % (
                score_sub11, select_sharpe, base_s_r['hs300']))
            else:
                print('评分：未得分。投资策略在投资周期的夏普比率为%0.2f%%，无法跑赢沪深300指数（%0.2f%%）。' % (
                select_sharpe, base_s_r['hs300']))
            if select_sharpe > base_s_r['csi500']:
                score_sub12 = self.score_dic['Comparison1'][1] + math.log(select_sharpe - base_s_r['csi500'] + 1)
                print('评分：得%0.4f分。投资策略在投资周期的夏普比率为%0.2f%%，可以跑赢中证500指数（%0.2f%%）。' % (
                score_sub12, select_sharpe, base_s_r['csi500']))
            else:
                print('评分：未得分。投资策略在投资周期的夏普比率为%0.2f%%，无法跑赢中证500指数（%0.2f%%）。' % (
                select_sharpe, base_s_r['csi500']))

            # 业绩基准的夏普比率
            rest_sharpe = 0
            for i in list(
                    set(self.select_dic.keys()).difference(set(self.hs300_list)).difference(set(self.csi500_list))):
                # 某行业
                industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
                # 某行业所有股票名
                stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values
                # 某行业的所有股票日线数据
                sharpe_value = 0
                for code in set(stock_list) & set(s_r.keys()):  # tqdm()
                    sharpe_value += s_r[code]
                rest_sharpe += (sharpe_value / len(set(stock_list) & set(s_r.keys()))) * prop[industry]

            base_sharpe = base_s_r['hs300'] * perc['沪深300'] + base_s_r['csi500'] * perc['中证500'] + \
                          rest_sharpe * perc['剩余股票']

            if select_sharpe > base_sharpe:
                score_sub13 = self.score_dic['Comparison1'][2] + math.log(select_sharpe - base_sharpe + 1)
                print('评分：得%0.4f分。投资策略在投资周期的夏普比率为%0.2f%%，可以跑赢业绩基准（%0.2f%%）。' % (
                score_sub13, select_sharpe, base_sharpe))
            else:
                print('评分：未得分。投资策略在投资周期的夏普比率为%0.2f%%，无法跑赢业绩基准（%0.2f%%）。' % (
                select_sharpe, base_sharpe))
            score_sub1 = score_sub11 + score_sub12 + score_sub13
        else:
            if select_sharpe > base_s_r['hs300']:
                score_sub11 += sum(self.score_dic['Comparison1']) / 2 + math.log(select_sharpe - base_s_r['hs300'] + 1)
                print('评分：得%0.4f分。投资策略在投资周期的夏普比率为%0.2f%%，可以跑赢沪深300指数（%0.2f%%）。' % (
                score_sub11, select_sharpe, base_s_r['hs300']))
            else:
                print('评分：未得分。投资策略在投资周期的夏普比率为%0.2f%%，无法跑赢沪深300指数（%0.2f%%）。' % (
                select_sharpe, base_s_r['hs300']))
            if select_sharpe > base_s_r['csi500']:
                score_sub12 = sum(self.score_dic['Comparison1']) / 2 + math.log(select_sharpe - base_s_r['csi500'] + 1)
                print('评分：得%0.4f分。投资策略在投资周期的夏普比率为%0.2f%%，可以跑赢中证500指数（%0.2f%%）。' % (
                score_sub12, select_sharpe, base_s_r['csi500']))
            else:
                print('评分：未得分。投资策略在投资周期的夏普比率为%0.2f%%，无法跑赢中证500指数（%0.2f%%）。' % (
                select_sharpe, base_s_r['csi500']))
            score_sub1 = score_sub11 + score_sub12

        print()
        print('基准对比：满分%0.4f，得分（含bonus）%0.4f。' % (sum(self.score_dic['Comparison1']), score_sub1))
        print()

        # 信息比率
        a_r = self.get_annualized_return(self.df_price)
        a_r_daily = self.get_return_daily(self.df_price)
        record_ar = 0
        record_ar_daily = 0
        for i in self.select_dic.keys():
            if i in a_r:
                # 年化收益率
                record_ar += a_r[i] * self.select_dic[i]
                # 每天的收益率
                record_ar_daily += a_r_daily[i] * self.select_dic[i]

        a_r_base = self.get_annualized_return(df_price_base)
        a_r_daily_base = self.get_return_daily(df_price_base)

        i_r_hs300 = self.get_information_ratio(record_ar, a_r_base['hs300'], record_ar_daily, a_r_daily_base['hs300'])
        i_r_csi500 = self.get_information_ratio(record_ar, a_r_base['csi500'], record_ar_daily,
                                                a_r_daily_base['csi500'])
        print('投资策略相对沪深300指数的信息比率为%0.2f。' % i_r_hs300)
        print('投资策略相对中证500指数的信息比率为%0.2f。' % i_r_csi500)

        if corr['投资组合']['业绩基准'] >= 0.70:
            rest_a_r = 0
            rest_a_r_daily = 0
            for i in list(
                    set(self.select_dic.keys()).difference(set(self.hs300_list)).difference(set(self.csi500_list))):
                # 某行业
                industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
                # 某行业所有股票名
                stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values
                # 某行业的所有股票日线数据
                a_r_value = 0
                a_r_value_daily = 0
                for code in set(stock_list) & set(a_r.keys()):
                    a_r_value += a_r[code]
                    a_r_value_daily += a_r_daily[code]
                rest_a_r += (a_r_value / len(set(stock_list) & set(a_r.keys()))) * prop[industry]
                rest_a_r_daily += (a_r_value_daily / len(set(stock_list) & set(a_r.keys()))) * prop[industry]

            base_return = a_r_base['hs300'] * perc['沪深300'] + a_r_base['csi500'] * perc['中证500'] + \
                          rest_a_r * perc['剩余股票']
            base_return_daily = a_r_daily_base['hs300'] * perc['沪深300'] + a_r_daily_base['csi500'] * perc['中证500'] + \
                                rest_a_r_daily * perc['剩余股票']
            i_r_base = self.get_information_ratio(record_ar, base_return, record_ar_daily, base_return_daily)
            print('投资策略相对业绩基准的信息比率为%0.2f。' % i_r_base)

        # ---------------------- 市场表现：同类股票 ----------------------#

        calender = [60, 90, 120, 180, 252]  # 可以自定义设置：近半个月、一个月、两个月、三个月、六个月、一年
        # temp = pro.daily(ts_code = [*self.select_dic.keys()][0], start_date = self.start_date, end_date = self.end_date_p)
        # 可以检验的近n天数
        pre_test = [i for i in calender if i <= len(self.df_price)]
        indi_score = self.score_dic['Comparison2'] * (1 / len(self.select_dic)) * (1 / len(pre_test))
        # print('小分值:%0.2f' %indi_score)

        print('-' * 90)
        rank_dic = {}
        for i in self.select_dic.keys():
            if i in self.Astock['ts_code'].values and i in a_r:
                # 所选股的所属行业
                industry = self.Astock[self.Astock['ts_code'] == i]['industry'].values[0]
                # 某行业所有股票名
                stock_list = self.Astock[self.Astock['industry'] == industry]['ts_code'].values
                days = []
                for day in pre_test:
                    temp1 = self.get_annualized_return(self.df_price[-day:])
                    temp2 = self.get_annualized_volatility(self.df_pctchg[-day:])
                    # 记录每个股票的年化收益、年化波动
                    record_ar = {}
                    record_av = {}
                    for code in set(stock_list) & set(temp1.keys()):
                        record_ar[code] = temp1[code]
                        record_av[code] = temp2[code]
                    # 所选股在同行业股票中top排名
                    s1 = dict(sorted(record_ar.items(), key=lambda item: item[1], reverse=True))
                    rank_ar = list(s1.keys()).index(i) + 1
                    s2 = dict(sorted(record_av.items(), key=lambda item: item[1]))
                    rank_av = list(s2.keys()).index(i) + 1
                    print('%s: 近%i天年化收益率位于同行业股票Top%i%%(%i/%i), 年化波动率位于同行业股票Top%i%% (%i/%i)。' \
                          % (
                          i, day, rank_ar / len(s1) * 100, rank_ar, len(s1), rank_av / len(s2) * 100, rank_av, len(s2)))

                    if int(rank_ar / len(s1) * 100) <= (100 - self.return_preference) and int(
                            rank_av / len(s2) * 100) <= self.risk_preference:
                        days.append(day)

                rank_dic[i] = days
        # print(indi_score,rank_dic)

        score_sub2 = 0
        for key in rank_dic.keys():
            score_sub2 += indi_score * len(rank_dic[key])

        score_sub2_bonus = 0
        if score_sub1 > 0 and score_sub2 > 0:
            score_sub2_bonus = self.bonus(score_sub2)

        print()
        print('同类股票对比：满分%0.4f，得分（含bonus）%0.4f。' % (
        self.score_dic['Comparison2'], score_sub2 + score_sub2_bonus))
        score = score_p1 + score_sub1 + score_sub2 + score_sub2_bonus
        print('-' * 90)
        print('-' * 90)
        print('收益风险评估总得分(含bonus)%0.4f。打分明细：%0.4f = %0.4f + %0.4f + %0.4f + %0.4f。' % (
        score, score, score_p1, score_sub1, score_sub2, score_sub2_bonus))
        print()
        print('用户偏好：%0.4f' % score_p1)
        print()
        print('基准对比：%0.4f' % score_sub1)
        print()
        print('同类股票对比：%0.4f' % score_sub2)
        print()
        print('同类股票对比bonus：%0.4f' % score_sub2_bonus)
        print('-' * 90)

        return df, round(score, 4)


class Get_Score(object):

    def __init__(self, select_dic, score_dic, return_preference, risk_preference, \
                 start_date, end_date, start_date_p, end_date_p, \
                 Astock_file, hs300_file, csi500_file):
        self.select_dic = select_dic
        self.score_dic = score_dic
        self.return_preference = return_preference
        self.risk_preference = risk_preference
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_p = start_date_p
        self.end_date_p = end_date_p
        self.Astock_file = Astock_file
        self.hs300_file = hs300_file
        self.csi500_file = csi500_file

    def load_stockcode(self, file_name):
        # 读取
        js = json.load(open(file_name, "r", encoding='utf-8'))
        list = [i['SECUCODE'] for i in js]
        return list

    def load_data(self, file_name):
        df = pd.read_csv(file_name)
        df = df.sort_values(by="trade_date")
        df.set_index('trade_date', inplace=True)
        df.index = pd.to_datetime(df.index, format='%Y%m%d').strftime('%Y-%m-%d')  # 有些时候需要
        df.index = pd.to_datetime(df.index)
        return df

    def cal_performance_assessment(self):

        for key in [*self.select_dic.keys()]:
            if self.select_dic[key] == 0:
                del self.select_dic[key]

        print(self.select_dic)
        # 读取

        js = json.load(open(self.Astock_file, "r", encoding='utf-8'))
        l = [[i['code'], i['行业'], i['股票简称']] for i in js]
        Astock = pd.DataFrame(l)
        Astock.columns = ['ts_code', 'industry', 'stock_name']
        Astock = Astock[Astock['industry'] != '-']

        industry_dic = {}
        for i in self.select_dic.keys():
            # 某行业
            industry = Astock[Astock['ts_code'] == i]['industry'].values[0]
            # 某行业所有股票名
            stock_list = Astock[Astock['industry'] == industry]['ts_code'].values
            print(i, industry, len(stock_list))
            # 某行业的所有股票日线数据
            industry_dic[industry] = industry_dic.get(industry, 0) + 1

        # 交易日历
        df_cal = pro.trade_cal(exchange='SZSE', start_date=self.start_date, end_date=self.end_date_p)
        data_num = len(df_cal[df_cal['is_open'] == 1])

        df_price = pd.DataFrame()
        df_pctchg = pd.DataFrame()
        for i in [*industry_dic.keys()]:
            # 某行业所有股票名
            stock_list = Astock[Astock['industry'] == i]['ts_code'].values
            for code in tqdm(stock_list):
                temp1 = pro.daily(ts_code=code, start_date=self.start_date, end_date=self.end_date_p)
                if temp1.shape[1] >= 3 and len(temp1) == data_num:  # 数据长度保持一致
                    temp = temp1[['trade_date', 'close', 'pct_chg']]
                    df_price[code] = temp['close']
                    df_pctchg[code] = temp['pct_chg']
                # time.sleep(12)
        df_price['trade_date'] = temp['trade_date']
        df_pctchg['trade_date'] = temp['trade_date']
        df_price.to_csv('df_price.csv', index=False)
        df_pctchg.to_csv('df_pctchg.csv', index=False)
        # print('Data save finish!')
        hs300_list = self.load_stockcode(self.hs300_file)
        csi500_list = self.load_stockcode(self.csi500_file)

        df_price = self.load_data('df_price.csv')
        df_pctchg = self.load_data('df_pctchg.csv')
        con = (df_price.index >= self.start_date) & (df_price.index <= self.end_date)
        # time.sleep(60)
        hs300 = pro.index_daily(ts_code='399300.SZ', start_date=self.start_date, end_date=self.end_date_p)
        csi500 = pro.index_daily(ts_code='399905.SZ', start_date=self.start_date, end_date=self.end_date_p)
        df_price_base = pd.DataFrame({'trade_date': hs300['trade_date'], 'hs300': hs300['close'], \
                                      'csi500': csi500['close']})
        df_price_base = df_price_base.sort_values(by="trade_date")
        df_price_base.set_index('trade_date', inplace=True)
        df_price_base.index = pd.to_datetime(df_price_base.index, format='%Y%m%d').strftime('%Y-%m-%d')
        df_price_base.index = pd.to_datetime(df_price_base.index)
        df_pctchg_base = pd.DataFrame({'trade_date': hs300['trade_date'], 'hs300': hs300['pct_chg'], \
                                       'csi500': csi500['pct_chg']})
        df_pctchg_base = df_pctchg_base.sort_values(by="trade_date")
        df_pctchg_base.set_index('trade_date', inplace=True)
        df_pctchg_base.index = pd.to_datetime(df_pctchg_base.index, format='%Y%m%d').strftime('%Y-%m-%d')
        df_pctchg_base.index = pd.to_datetime(df_pctchg_base.index)

        A = Assessment(self.select_dic, self.score_dic, self.return_preference, self.risk_preference, \
                       self.start_date, self.end_date, self.start_date_p, self.end_date_p, \
                       Astock, hs300_list, csi500_list, \
                       df_price[con], df_pctchg[con], df_price_base, df_pctchg_base)
        performance_assessment_results_dict = A.cal_performance_assessment(df_price_base, df_pctchg_base)
        return performance_assessment_results_dict


def min_max_normalize(scores_dict):
    # 获取最大和最小得分
    max_score = max(scores_dict.values())
    min_score = min(scores_dict.values())

    if min_score == max_score == 0:
        return scores_dict

    else:

        # 最大最小归一化
        normalized_scores = {key: (value - min_score) / (max_score - min_score) for key, value in scores_dict.items()}

        return normalized_scores


