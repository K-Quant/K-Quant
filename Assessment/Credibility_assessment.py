import argparse
import json
import os

from Assessment.metrics import cal_assessment
from run_assessment import main


def run_credibility_assessment(param_dict, data_loader, model, explanation_model):
    reliability, stability, explainable, robustness, transparency = \
        cal_assessment(param_dict, data_loader, model, explanation_model, "cpu")

    credibility_assessment_results_dict ={
        "可靠性得分": reliability,
        "稳定性得分": stability,
        "鲁棒性得分": robustness,
        "透明性得分": transparency,
        "解释效果得分": explainable
    }
    return credibility_assessment_results_dict





