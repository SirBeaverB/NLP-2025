import os
import json
import glob
import yaml
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch
from torch.nn.parallel import DataParallel

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config['data_path']
INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']

# 加载语料
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
                text = data.get('chunk_content') or data.get('text')
                title = data.get('title', '')
                full_text = (title or '') + ' ' + (text or '')
                if text:
                    corpus.append(data)
                    contents.append(full_text)
            elif isinstance(data, str):
                corpus.append({'chunk_content': data})
                contents.append(data)
print(f"Found {len(contents)} contents in {DATA_PATH}")



# 生成embedding
model = SentenceTransformer(EMBEDDING_MODEL)
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs for embedding (DataParallel)")
        model = DataParallel(model)
        device = 'cuda'
    else:
        print('Using single GPU for embedding')
        device = 'cuda'
    model = model.to(device)
else:
    print('Using CPU for embedding')
    device = 'cpu'

# DataParallel下需特殊处理encode
if isinstance(model, DataParallel):
    def dp_encode(texts, **kwargs):
        # DataParallel下forward返回list,需拼接
        all_emb = []
        batch_size = kwargs.get('batch_size', 32)
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            emb = model.module.encode(batch, device=device, convert_to_numpy=True)
            all_emb.append(emb)
        return np.concatenate(all_emb, axis=0)
    embeddings = dp_encode(contents, batch_size=64)
else:
    embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True, device=device)
if isinstance(embeddings, np.ndarray):
    embeddings = torch.from_numpy(embeddings)
embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
if hasattr(embeddings, 'cpu'):
    embeddings = embeddings.cpu().numpy()
    
# 构建faiss索引
if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
    ngpus = faiss.get_num_gpus()
    print(f"Using {ngpus} GPUs for FAISS indexing")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)
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