import os
import json
import glob
import yaml
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config['data_path']
INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']

# 加载语料
files = glob.glob(os.path.join(DATA_PATH, '*.json'))
print(f"Found {len(files)} JSON files in {DATA_PATH}")
corpus = []
contents = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        if isinstance(data_list, dict):
            data_list = [data_list]
        for data in data_list:
            if isinstance(data, dict):
                text = data.get('content') or data.get('text')
                if text:
                    corpus.append(data)
                    contents.append(text)
            elif isinstance(data, str):
                corpus.append({'content': data})
                contents.append(data)
print(f"Found {len(contents)} contents in {DATA_PATH}")


# 生成embedding
model = SentenceTransformer(EMBEDDING_MODEL)
if torch.cuda.is_available():
    model = model.to('cuda')
    print('Using CUDA for embedding')
else:
    print('Using CPU for embedding')
embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# 构建faiss索引
if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)
    print('Using FAISS GPU for indexing')
else:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print('Using FAISS CPU for indexing')

# 保存索引和语料
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(index, INDEX_PATH)
with open(INDEX_PATH + '.meta', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False)

print(f"索引已保存到 {INDEX_PATH}，共 {len(corpus)} 条文档。") 