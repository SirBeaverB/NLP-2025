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

# 加载mREBEL模型和分词器
model_name = "Babelscape/mrebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="zh_CN", tgt_lang="tp_XX")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def extract_triplets_typed(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '', '', '', '', ''
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__zh__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append((subject.strip(), relation.strip(), object_.strip()))
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets

corpus_dir = 'data'
out_dir = 'kg/triples'
os.makedirs(out_dir, exist_ok=True)

# 每个文件存储的最大三元组数
MAX_TRIPLES_PER_FILE = 1000

def write_triples_to_file(triples, file_idx):
    """将三元组写入指定文件"""
    out_file = os.path.join(out_dir, f'triples_{file_idx:03d}.csv')
    with open(out_file, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['head', 'relation', 'tail'])
        for h, r, t in triples:
            writer.writerow([h, r, t])
    return out_file

# 初始化计数器
current_file_idx = 0
current_triples = []
# 新增：全局三元组去重集合
seen_triples = set()

files = glob.glob(os.path.join(corpus_dir, '*.json'))
for file in tqdm(files, desc="处理文件"):
    try:
        with open(file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            # 处理单个文档
            if isinstance(data, dict):
                if 'content' in data:
                    text = data['content']
                    if text:
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                        outputs = model.generate(
                            **inputs,
                            max_length=256,
                            num_beams=3,
                            num_return_sequences=1,
                            decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX")
                        )
                        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                        triples = extract_triplets_typed(decoded[0])
                        # 去重：只添加未出现过的三元组
                        for triple in triples:
                            if triple not in seen_triples:
                                current_triples.append(triple)
                                seen_triples.add(triple)
                        # 检查是否需要写入文件
                        if len(current_triples) >= MAX_TRIPLES_PER_FILE:
                            out_file = write_triples_to_file(current_triples, current_file_idx)
                            print(f'已保存第 {current_file_idx+1} 批三元组到 {out_file}')
                            current_triples = []
                            current_file_idx += 1
            # 处理文档列表
            elif isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        text = doc.get('content', '')
                        if text:
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                            outputs = model.generate(
                                **inputs,
                                max_length=256,
                                num_beams=3,
                                num_return_sequences=1,
                                decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX")
                            )
                            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                            triples = extract_triplets_typed(decoded[0])
                            # 去重：只添加未出现过的三元组
                            for triple in triples:
                                if triple not in seen_triples:
                                    current_triples.append(triple)
                                    seen_triples.add(triple)
                            # 检查是否需要写入文件
                            if len(current_triples) >= MAX_TRIPLES_PER_FILE:
                                out_file = write_triples_to_file(current_triples, current_file_idx)
                                print(f'已保存第 {current_file_idx+1} 批三元组到 {out_file}')
                                current_triples = []
                                current_file_idx += 1
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")
        continue

# 写入剩余的三元组
if current_triples:
    out_file = write_triples_to_file(current_triples, current_file_idx)
    print(f'已保存最后一批三元组到 {out_file}')

print(f'所有三元组已分批保存到 {out_dir} 目录')