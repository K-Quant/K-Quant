# ---------------------
# get all paper link of coog
# ---------------------
from unicodedata import bidirectional
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bf
import re
import json
from bs4.element import CharsetMetaAttributeValue
from tqdm import tqdm
import time
import random
import tushare as ts
import ast
import datetime
import requests
import akshare as ak
import sys
import argparse

sys.path.append('./company_rel')
sys.path.append('./DART')
# import demo
from demo import run_extraction

from dart_pipline import fusion

# -----------------------------------------------

NOT_FOUND_CODE = 100001
NOT_FOUND_MSG = "news not found"

SUCCESS_CODE = 100000

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
cookies = {
    '__auc=e338969417ef7344ae449919c12; __asc=4cb7c69617f201cbce9bf466dca; __utma=177965731.1778915295.1644824906.1644824906.1645511229.2; __utmc=177965731; __utmz=177965731.1645511229.2.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); CookiePolicyCheck=0; _cc_id=af14ad876611b8e47d20db0ad43c1052; __utma=81143559.738177882.1644824906.1644824906.1645511246.2; __utmc=81143559; __utmz=81143559.1645511246.2.2.utmcsr=aastocks.com|utmccn=(referral)|utmcmd=referral|utmcct=/; NewsZoomLevel=3; ASP.NET_SessionId=bt3zwgrxrakas3zczxxsv4bi; mLang=TC; AALive=3429; cto_bundle=rXZ9LV8lMkIwZElmOVNvWTFyd2lDNGdIM3FiaWtwSVBJcXVxMW4xeDk3SDNHdE9ZNzNLVCUyRnhJJTJGeUE5a2xkTklBTzRiWVBQUFhLT0g3R1RRYjBVanFVbVBwM0RlR2lNN0hFcU5KOFBLcHZ5SyUyRlZUR3dmaVMxOE0yZE1NNHZjMk5adXpPbEFSdHlqQ0M3b01sQnMxOWowZ3lHQXJqQSUzRCUzRA; __gads=ID=ee522e9dfd1d4c35-22c82040b6d0004c:T=1644824906:RT=1645515431:S=ALNI_MZWY7mcN-MhVR0tdkFGR1Uu3F4XYA; aa_cookie=218.102.252.238_62750_1645513384; __utmt_a3=1; __utmb=177965731.71.9.1645515545042; __utmt_a2=1; __utmt_b=1; __utmb=81143559.153.9.1645515500316'
}


def returnN(x):
    return None if len(x) == 0 else x


def get_tushare_news_n_days(start_date_str, required_days):
    ts.set_token() # please fill the Tushare token in the bracket to use these function
    pro = ts.pro_api()
    datetime_str = start_date_str + ' 00:00:00'
    today = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')
    result = []
    # today = datetime.datetime.today()
    for i in ['10jqka', 'eastmoney', 'yuncaijing']:
        news_list = []
        for t in tqdm(range(0, required_days)):
            end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d') + ' 12:00:00'
            start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d') + ' 00:00:00'
            df = pro.news(src=i, start_date=start_day, end_date=end_day)
            news_list.extend(df.to_dict("records"))

            end_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d') + ' 23:59:59'
            start_day = (today - datetime.timedelta(days=t)).strftime('%Y-%m-%d') + ' 12:00:01'
            df = pro.news(src=i, start_date=start_day, end_date=end_day)
            news_list.extend(df.to_dict("records"))

        for news in news_list:
            news['source'] = i
        result.extend(news_list)
    if result:
        print("Recent "+str(required_days)+" days news from " + start_date_str + "has been saved.")
        # print(result)
        write_list_to_json(result,'./crawl_data/crawl_for_extraction.json')
    #     return SUCCESS_CODE, result
    # else:
    #     return NOT_FOUND_CODE, NOT_FOUND_MSG

def write_list_to_json(list,json_file_name):
    with open(json_file_name,'w') as f:
        json.dump(list,f)

def dynamic_pipline(start_date_str, required_days, original_news_path, extracted_data_path, input_format, fusion_data_path):
    get_tushare_news_n_days(start_date_str, required_days)
    run_extraction(original_news_path, extracted_data_path, input_format)
    df = fusion(extracted_data_path)
    df.to_csv(fusion_data_path)
    jdata = df.to_json(orient='records', force_ascii=False)
    return jdata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_date", type=str, default=None,
                        help="The start date of news you want to extract knowledge from.")
    parser.add_argument("--required_days", type=str, default=None,
                        help="The required days interval.")
    parser.add_argument("--original_news_path", type=str, default=None,
                        help="The path of input text data that you want to load.")
    parser.add_argument("--extracted_data_path", type=str, default=None,
                        help="The path of extracted knowledge that you want to save in.")
    parser.add_argument("--input_format", type=str, default=None,
                        help="The input file format, such as json.")
    parser.add_argument("--fusion_data_path", type=str, default=None,
                        help="The path of fusion knowledge that you want to save in.")
    args = parser.parse_args()

    dynamic_pipline(args.start_date, args.required_days, args.original_news_path, args.extracted_data_path, args.input_format, args.fusion_data_path)
