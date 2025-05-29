import os
import glob
import json
import csv
from tqdm import tqdm
import math
import torch
from ltp import LTP

"""
从词性、命名实体识别、依存句法分析、语义角色标注中手动提取三元组还是太过困难。遂放弃。
此为废稿，仅作记录，以备后续参考。
"""


ltp = LTP("LTP/small")  # 默认加载 Small 模型

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
# 使用字典格式作为返回结果
print(output.cws)  # [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
print(output.pos) # [['r', 'v', 'nr', 'v', 'v', 'n', 'wp']]
print(output.dep) # [{'head': [2, 0, 2, 2, 4, 5, 2], 'label': ['AGT', 'Root', 'DATV', 'eSUCC', 'eSUCC', 'PAT', 'mPUNC']}]
print(output.ner) # [[('Nh', '汤姆', 2, 2)]]
print(output.srl) # [[{'index': 1, 'predicate': '叫', 'arguments': [('A0', '他', 0, 0), ('A1', '汤姆', 2, 2), ('A2', '去拿外衣', 3, 5)]}, {'index': 4, 'predicate': '拿', 'arguments': [('A0', '汤姆', 2, 2), ('A1', '外衣', 5, 5)]}]]


# 使用感知机算法实现的分词、词性和命名实体识别，速度比较快，但是精度略低
ltp = LTP("LTP/legacy")
# cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "ner"]).to_tuple() # error: NER 需要 词性标注任务的结果
cws, pos, ner = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner"]).to_tuple()  # to tuple 可以自动转换为元组格式
# 使用元组格式作为返回结果
print(cws, pos, ner)

corpus_dir = 'data'
out_dir = 'kg/triples_ltp'
os.makedirs(out_dir, exist_ok=True)
MAX_TRIPLES_PER_FILE = 10000

ltp = LTP()

# NER类型映射：人名（Nh）、地名（Ns）、机构名（Ni）
NER_TYPES = {"Nh", "Ns", "Ni"}

def merge_phrases(words, pos, target_prefix):
    merged = []
    indices = []
    i = 0
    while i < len(words):
        if pos[i].startswith(target_prefix):
            start = i
            phrase = words[i]
            i += 1
            while i < len(words) and pos[i].startswith(target_prefix):
                phrase += words[i]
                i += 1
            merged.append(phrase)
            indices.append((start, i-1))
        else:
            merged.append(words[i])
            indices.append((i, i))
            i += 1
    return merged, indices

def find_phrase(index, indices, phrases):
    for idx, (start, end) in enumerate(indices):
        if start <= index <= end:
            return phrases[idx]
    return None

def extract_triples(text):
    triples = []
    output = ltp.pipeline([text], tasks=["cws", "pos", "ner", "srl"])
    cws = output.cws[0]
    pos = output.pos[0]
    ner = output.ner[0]
    srl = output.srl[0]
    # 构建NER实体区间列表
    ner_spans = []
    for ner_item in ner:
        if len(ner_item) == 4:
            ent_type, entity, start, end = ner_item
        elif len(ner_item) == 3:
            ent_type, start, end = ner_item
            entity = ''.join(cws[start:end+1])
        else:
            continue
        ner_spans.append((start, end, entity))
    # 合并名词短语
    noun_phrases, noun_indices = merge_phrases(cws, pos, 'n')
    def get_entity_or_phrase(start, end):
        # 优先NER实体
        for s, e, entity in ner_spans:
            if s >= start and e <= end:
                return entity
        # 否则用合并名词短语
        for idx, (s, e) in enumerate(noun_indices):
            if s <= start and end <= e:
                return noun_phrases[idx]
        # 否则用原始短语
        return ''.join(cws[start:end+1])
    for pred in srl:
        predicate = pred['predicate']
        args = {role: (start, end) for role, _, start, end in pred['arguments']}
        if 'A0' in args and 'A1' in args:
            s1, e1 = args['A0']
            s2, e2 = args['A1']
            subj = get_entity_or_phrase(s1, e1)
            obj = get_entity_or_phrase(s2, e2)
            triples.append((subj, predicate, obj))
        elif 'A0' in args and 'A2' in args:
            s1, e1 = args['A0']
            s2, e2 = args['A2']
            subj = get_entity_or_phrase(s1, e1)
            obj = get_entity_or_phrase(s2, e2)
            triples.append((subj, predicate, obj))
    return triples

def write_triples_to_file(triples, file_idx):
    out_file = os.path.join(out_dir, f'triples_{file_idx:03d}.csv')
    with open(out_file, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['head', 'relation', 'tail'])
        for h, r, t in triples:
            writer.writerow([h, r, t])
    return out_file

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
                    if text:
                        triples = extract_triples(text)
                        print(f"\n从文件 {file} 提取的三元组:")
                        print(f"三元组数量: {len(triples)}")
                        for h, r, t in triples:
                            print(f"头实体: {h} | 关系: {r} | 尾实体: {t}")
                        all_triples.extend(triples)
            # 处理文档列表
            elif isinstance(data, list):
                for doc in data:
                    if isinstance(doc, dict):
                        text = doc.get('content', '')
                        if text:
                            triples = extract_triples(text)
                            print(f"\n从文件 {file} 提取的三元组:")
                            print(f"三元组数量: {len(triples)}")
                            for h, r, t in triples:
                                print(f"头实体: {h} | 关系: {r} | 尾实体: {t}")
                            all_triples.extend(triples)
    except Exception as e:
        print(f"处理文件 {file} 时出错: {str(e)}")
        continue

print(f"\n总共提取的三元组数量: {len(all_triples)}")
print("\n所有三元组示例:")
for i, (h, r, t) in enumerate(all_triples[:10]):  # 只显示前10个三元组作为示例
    print(f"{i+1}. 头实体: {h} | 关系: {r} | 尾实体: {t}")

total_files = math.ceil(len(all_triples) / MAX_TRIPLES_PER_FILE)
for i in range(total_files):
    start_idx = i * MAX_TRIPLES_PER_FILE
    end_idx = min((i + 1) * MAX_TRIPLES_PER_FILE, len(all_triples))
    batch_triples = all_triples[start_idx:end_idx]
    out_file = write_triples_to_file(batch_triples, i)
    print(f'已保存第 {i+1}/{total_files} 批三元组到 {out_file}')

print(f'所有三元组已分批保存到 {out_dir} 目录')