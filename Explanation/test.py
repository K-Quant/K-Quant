import json

# 打开JSON文件
with open('outputDataNRSRRank.json', 'r') as f:
    data = json.load(f)

# 输出读取到的数据
print(data['2022-01-04']['SH600085'])
print(len(data['2022-01-04'].keys()))