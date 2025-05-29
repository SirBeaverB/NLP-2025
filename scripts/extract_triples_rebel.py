from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import glob
import os
import json
import csv
from tqdm import tqdm
import math

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载REBEL模型和分词器
model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

corpus_dir = 'data'
out_dir = 'kg/triples'
os.makedirs(out_dir, exist_ok=True)

# 每个文件存储的最大三元组数
MAX_TRIPLES_PER_FILE = 10000

# rebel输出的三元组解析函数
def parse_rebel_output(text):
    triples = []
    # rebel输出格式: <triplet> subj <subj> rel <rel> obj <obj>
    for chunk in text.split('<triplet>'):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            subj = chunk.split('<subj>')[1].split('<rel>')[0].strip()
            rel = chunk.split('<rel>')[1].split('<obj>')[0].strip()
            obj = chunk.split('<obj>')[1].strip()
            triples.append((subj, rel, obj))
        except Exception:
            continue
    return triples

def write_triples_to_file(triples, file_idx):
    """将三元组写入指定文件"""
    out_file = os.path.join(out_dir, f'triples_{file_idx:03d}.csv')
    with open(out_file, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['head', 'relation', 'tail'])
        for h, r, t in triples:
            writer.writerow([h, r, t])
    return out_file

# 收集所有三元组
all_triples = []
files = glob.glob(os.path.join(corpus_dir, '*.json'))
for file in tqdm(files, desc="处理文件"):
    try:
        with open(file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            # 处理单个文档
            if isinstance(data, dict):
                if 'content' in data:
                    text = data['content']
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    outputs = model.generate(**inputs, max_length=512)
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    triples = parse_rebel_output(decoded[0])
                    all_triples.extend(triples)
            # 处理文档列表
            elif isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        text = doc.get('content', '')
                        if text:
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                            outputs = model.generate(**inputs, max_length=512)
                            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                            triples = parse_rebel_output(decoded[0])
                            all_triples.extend(triples)
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")
        continue

# 将三元组分批写入文件
total_files = math.ceil(len(all_triples) / MAX_TRIPLES_PER_FILE)
for i in range(total_files):
    start_idx = i * MAX_TRIPLES_PER_FILE
    end_idx = min((i + 1) * MAX_TRIPLES_PER_FILE, len(all_triples))
    batch_triples = all_triples[start_idx:end_idx]
    out_file = write_triples_to_file(batch_triples, i)
    print(f'已保存第 {i+1}/{total_files} 批三元组到 {out_file}')

print(f'所有三元组已分批保存到 {out_dir} 目录')