import ast
import re
from data_processing import *
import os

dirname, filename = os.path.split(os.path.abspath(__file__))


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()
    return para.split("\n")


def split_sen(s, name_dict, code_dict, src="default", time="default"):
    if len(name_dict) != len(code_dict):
        print('length of name and code is different, pls check')
    if s != None:
        lis = s.split("。")
        res = []
        for sen in lis:
            count = 0
            h = None
            t = None
            for name in name_dict:
                if sen.find(name) != -1:
                    count += 1
                    if count == 1:
                        h = name
                    elif count == 2:
                        if sen.find(name) > sen.find(h):
                            t = name
                        else:
                            t = h
                            h = name
            if count == 2 and h != t:
                codeh = code_dict[name_dict.index(h)]
                codet = code_dict[name_dict.index(t)]
                if time != "default" and src != "default":
                    temp = {
                        'head': h,
                        'tail': t,
                        'sentence': sen,
                        'time': time,
                        'head_code': codeh,
                        'tail_code': codet,
                        'src': src
                    }
                else:
                    temp = {
                        'head': h,
                        'tail': t,
                        'time': time,
                        'src': src,
                        'sentence': sen,
                        'head_code': codeh,
                        'tail_code': codet,
                    }
                res.append(temp)

    else:
        return []
    return res


def sentence_set(raw_file_list):
    name_dict = []
    code_dict = []
    file = open(dirname + '/data/company_full_name.json')
    for line in file:
        x = ast.literal_eval(line)
        name_dict.append(x['name'])

    file = open(dirname + '/data/company_full_name.json')
    for line in file:
        x = ast.literal_eval(line)
        code_dict.append(x['code'])

    total_set = []
    dicct = json.load(open(raw_file_list))
    for i in tqdm(dicct):
        sentence = i['content']
        time = i['datetime']
        src = i["source"]
        total_set.extend(split_sen(sentence, name_dict, code_dict, src, time))
    print(len(total_set))


    json.dump(total_set, open(dirname + '/data/news_sententces_set.json', 'w', encoding='utf8'),
              indent=1, ensure_ascii=False)


def user_input_parse(file_type, raw_file_list, input_mode="default"):
    if input_mode == "default":
        sentence_set(raw_file_list)
        return True

    name_dict = []
    code_dict = []
    file = open(dirname + '/data/company_full_name.json')
    for line in file:
        x = ast.literal_eval(line)
        name_dict.append(x['name'])

    file = open(dirname + '/data/company_full_name.json')
    for line in file:
        x = ast.literal_eval(line)
        code_dict.append(x['code'])

    new_file_list = []
    if file_type == "pdf":
        for name in raw_file_list:
            out_name = name[:-3] + "txt"
            new_file_list.append(out_name)
            parse_PDF2txt(name, out_name)
        raw_file_list = new_file_list
        file_type = "txt"

    if file_type == "html_url":
        num = 1
        for name in raw_file_list:
            out_name = str(num) + ".txt"
            new_file_list.append(out_name)
            parse_htmlURL2json(name)
            get_content(out_name)
            num += 1
        raw_file_list = new_file_list
        file_type = "txt"

    if file_type == "txt":
        total_set = []
        cutted_list = []
        for name in raw_file_list:
            sentences = list(pd.read_table(name, header=None)[0])
            for sentence in sentences:
                for item in cut_sent(sentence):
                    cutted_list.append(item)
        for sen in cutted_list:
            total_set.extend(split_sen(sen, name_dict, code_dict))

    if len(total_set) == 0:
        return False
    json.dump(total_set, open(dirname + '/data/news_sententces_set.json', 'w', encoding='utf8'),
              indent=1, ensure_ascii=False)

    return True
