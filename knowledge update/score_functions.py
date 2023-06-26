import numpy as np


def score1(rule, c=0):
    """
    Calculate candidate score depending on the rule's confidence.

    Parameters:
        rule (dict): rule from rules_dict
        c (int): constant for smoothing

    Returns:
        score (float): candidate score
    """

    score = rule["rule_supp"] / (rule["body_supp"] + c)

    return score


def score2(cands_walks, test_query_ts, lmbda):
    """
    Calculate candidate score depending on the time difference.

    Parameters:
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution

    Returns:
        score (float): candidate score
    """

    max_cands_ts = max(cands_walks["timestamp_0"])
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)
    )  # Score depending on time difference

    return score


def score_12(rule, cands_walks, test_query_ts, lmbda, a):
    """
    Combined score function.

    Parameters:
        rule (dict): rule from rules_dict
        cands_walks (pd.DataFrame): walks leading to the candidate
        test_query_ts (int): test query timestamp
        lmbda (float): rate of exponential distribution
        a (float): value between 0 and 1

    Returns:
        score (float): candidate score
    """

    score = a * score1(rule) + (1 - a) * score2(cands_walks, test_query_ts, lmbda)

    return score
