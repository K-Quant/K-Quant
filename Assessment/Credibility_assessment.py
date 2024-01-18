import argparse
import json
import os

from run_assessment import main, test_get_stocks_recommendation





if __name__ == '__main__':
    args = parse_args()
    print(test_get_stocks_recommendation(args))

