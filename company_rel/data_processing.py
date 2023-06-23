import codecs
import sys
import threading
from queue import Queue

import pandas as pd
from tqdm import tqdm

sys.path.append('Parsley')
from Parsley.http_parser.master_parser import MasterParser
from Parsley.tools.general import *
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
import os

dirname, filename = os.path.split(os.path.abspath(__file__))

# The mapping dictionary
dicct = {'1': '增持',
         '2': '被增持',
         '3': '减持',
         '4': '被减持',
         '5': '供应',
         '6': '被供应',
         '7': '投资',
         '8': '被投资',
         '9': '上级',
         '10': '下级',
         '11': '同跌',
         '12': '同涨',
         '13': '合作',
         '14': '竞争',
         '15': '同行',
         '16': '纠纷',
         '0': '未知Unknown'
         }

INPUT_FILE = dirname + '/Parsley/sample-links.txt'
OUTPUT_DIR = dirname + '/Parsley/data'
NUMBER_OF_THREADS = 8

queue = Queue()
crawl_count = 0


def remove_blanks(df):
    for idx in tqdm(range(len(df["sentence"]))):
        while ' ' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace(' ', '，')
        while ' ' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace(' ', '')
        while '   ' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace('   ', '，')
        while '　' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace('　', '，')
        while '\n' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace('\n', '，')
        while '    ' in df["sentence"][idx]:
            df["sentence"][idx] = df["sentence"][idx].replace('    ', '，')
    return df


def get_pose(text, E_name):
    result = []
    pose_1 = text.find(E_name)
    pose_2 = pose_1 + len(E_name) - 1
    result.append(pose_1)
    result.append(pose_2)
    return result


def prepare_company_relation(original_test_json_path, input_mode):
    with open(original_test_json_path, "r") as f:
        sentences_dict = json.load(f)
    sentences_df = pd.DataFrame(sentences_dict)
    sentences_df = remove_blanks(sentences_df)
    sentences_df.to_csv(dirname + '/data/final_merge.csv', index=False, mode="w")
    if input_mode == "default":
        sentences_df = sentences_df.drop(columns=['time', 'head_code', 'tail_code', 'src'])
    else:
        sentences_df = sentences_df.drop(columns=['head_code', 'tail_code'])

    sentences_df.to_csv(dirname + '/data/company_relation.txt', sep='\t', header=None, index=False, mode="w")
    return


def prepare_dataset(divide_label=False):
    f = open(dirname + '/data/comp_pred.txt', 'w+')

    if divide_label:
        f1 = open(dirname + '/data/comp_train.txt', "w+")
        f2 = open(dirname + '/data/comp_test.txt', 'w+')

    # Should firstly run function "prepare_company_relation"
    company_relation = dirname + "/data/company_relation.txt"

    rel_set = []
    with codecs.open(company_relation, 'r', 'utf-8') as tfc:
        for lines in tfc:
            line = lines.split()
            data = {}
            if divide_label:
                if len(line) != 4:
                    continue
                E1, E2, relation, text = line
            else:
                # if len(line) != 3:
                #   print(line)
                #    continue
                if len(line) != 6:
                    print(line)
                    continue
                E1, E2, text, time, time2, src = line
            data["token"] = list(text)
            h = {}
            h["name"] = E1
            h["pos"] = get_pose(text, E1)
            data["h"] = h
            t = {}
            t["name"] = E2
            t["pos"] = get_pose(text, E2)
            data["t"] = t
            if divide_label:
                data["relation"] = relation
            else:
                # When do not need to train a model,
                # the dataset "relation" before relationship prediction model should be initialized as "None".
                data["relation"] = "未知Unknown"
            rel_set.append(data)
        if divide_label:
            rel_train = rel_set[:int(len(rel_set) * 0.8)]
            rel_test = rel_set[int(len(rel_set) * 0.8):]
            for x in rel_train:
                json_data = json.dumps(x, ensure_ascii=False)
                f1.write(json_data)
                f1.write("\n")
            for x in rel_test:
                json_data = json.dumps(x, ensure_ascii=False)
                f2.write(json_data)
                f2.write("\n")
        else:
            rel_pred = rel_set
        for x in rel_pred:
            json_data = json.dumps(x, ensure_ascii=False)
            f.write(json_data)
            f.write("\n")
    return


def obtain_final_predictions(pred_result_json_path, save_csv_path):
    company_df = pd.read_csv(dirname + "/data/final_merge.csv")
    print(len(company_df))
    # read predicted results
    with open(pred_result_json_path, "r") as f:
        pred_list = json.load(f)
    print(len(pred_list))
    if len(pred_list) == len(company_df):
        company_df["relation"] = pred_list
        company_df['relation'] = company_df['relation'].apply(lambda x: dicct[str(x)])
        company_df.to_csv(save_csv_path, index=False, mode="w")
    print(company_df)
    return


def create_workers():
    for _ in range(NUMBER_OF_THREADS):
        t = threading.Thread(target=work)
        t.daemon = True
        t.start()


def work():
    global crawl_count
    while True:
        url = queue.get()
        crawl_count += 1
        MasterParser.parse(url, OUTPUT_DIR, str(crawl_count))
        queue.task_done()


def create_jobs():
    for url in file_to_set(INPUT_FILE):
        queue.put(url)
    queue.join()


def parse_htmlURL2json(url):
    with open(dirname + '/Parsley/sample-links.txt', 'w') as f:
        f.write(url)
    create_dir(OUTPUT_DIR)
    create_workers()
    create_jobs()
    return


def parse_PDF2txt(file_path, out_path):
    fp = open(file_path, 'rb')
    # 创建一个与文档相关的解释器
    parser = PDFParser(fp)

    # pdf文档的对象，与解释器连接起来
    doc = PDFDocument(parser=parser)
    parser.set_document(doc=doc)

    # 如果是加密pdf，则输入密码
    # doc._initialize_password()

    # 创建pdf资源管理器
    resource = PDFResourceManager()

    # 参数分析器
    laparam = LAParams()

    # 创建一个聚合器
    device = PDFPageAggregator(resource, laparams=laparam)

    # 创建pdf页面解释器
    interpreter = PDFPageInterpreter(resource, device)

    # 获取页面的集合
    for page in PDFPage.get_pages(fp):
        # 使用页面解释器来读取
        interpreter.process_page(page)

        # 使用聚合器来获取内容
        layout = device.get_result()
        for out in layout:
            if hasattr(out, 'get_text'):
                # 写入txt文件
                fw = open(out_path, 'a')
                fw.write(out.get_text())
    return


def get_content(out_path):
    with open(dirname + "/Parsley/data/1.json", 'r') as load_f:
        load_dict = json.load(load_f)
    content_list = list()
    for item in load_dict["tags"]:
        content = item["content"]
        if content != None and len(content) > 8 and len(content) < 500:
            content_list.append(content)
    with open(out_path, 'w') as f:
        for line in content_list:
            f.write(line + '\n')

    return

# parse_htmlURL2json("https://www.eastmoney.com/")
# parse_PDF2txt("pdftest.pdf")
# get_content()
