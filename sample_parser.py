import re
import json

def parse_document_to_json(text):
    samples = []
    current_sample = {}
    capturing = None
    buffer = []

    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 检测样本开始
        if line.startswith('样本'):
            if current_sample:
                samples.append(current_sample)
                current_sample = {}
            current_sample["index"] = int(line.replace('样本', '').replace(':', '').strip())
            continue

        # 检测字段开始
        for field in ['题目ID', '时间序列', '题目类别', '标签答案']:
            if line.startswith(field):
                capturing = field
                buffer = [re.search(r'\[(.*)', line).group(1)]  # 捕获首行内容
                continue

        # 数据收集
        if capturing:
            if ']' in line:
                # 结束捕获
                buffer.append(line.replace(']', '').strip())
                full_content = ' '.join(buffer)
                
                # 提取所有数字
                numbers = list(map(int, re.findall(r'\d+', full_content)))
                
                # 根据字段存储
                field_map = {
                    '题目ID': 'input_ids',
                    '时间序列': 'input_rtime',
                    '题目类别': 'input_cat',
                    '标签答案': 'labels'
                }
                current_sample[field_map[capturing]] = numbers
                capturing = None
                buffer = []
            else:
                buffer.append(line.strip())

        # 检测样本结束
        if line.startswith('--------------------------------------------------'):
            if current_sample:
                samples.append(current_sample)
                current_sample = {}

    # 处理最后一个样本
    if current_sample:
        samples.append(current_sample)

    return {"samples": samples}