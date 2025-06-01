import os
import json

def split_into_chunks(text, chunk_size=400, overlap=200):
    """将文本切成多个chunk，支持重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

data_dir = 'data'
output_dir = 'data_chunked'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if not filename.endswith('.json'):
        continue
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(output_dir, filename)
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
        if isinstance(data_list, dict):
            data_list = [data_list]
    chunked_list = []
    for data in data_list:
        content = data.get('content') or data.get('text')
        if not content:
            continue
        chunks = split_into_chunks(content, chunk_size=300, overlap=50)
        for idx, chunk in enumerate(chunks):
            chunk_data = data.copy()  # 保留原有元信息
            chunk_data['chunk_content'] = chunk
            chunk_data['chunk_id'] = idx
            chunked_list.append(chunk_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_list, f, ensure_ascii=False, indent=2)

print("切块完成，结果保存在 NLP-2025/data_chunked/")
