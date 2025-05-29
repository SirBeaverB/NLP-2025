import os
import json
import glob
import yaml
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config['data_path']
INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']

# 加载语料
files = glob.glob(os.path.join(DATA_PATH, '*.json'))
corpus = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = data.get('content') or data.get('text')
                    if text:
                        corpus.append(text)
                elif isinstance(data, str):
                    corpus.append(data)
            except Exception:
                continue

# 生成embedding
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

# 构建faiss索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 保存索引和语料
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)
with open(INDEX_PATH + '.meta', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False)

print(f"索引已保存到 {INDEX_PATH}，共 {len(corpus)} 条文档。") 