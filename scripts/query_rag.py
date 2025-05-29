import os
import json
import yaml
import faiss
import openai
from sentence_transformers import SentenceTransformer

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']
TOP_K = config['top_k']

# 加载faiss索引和元数据
index = faiss.read_index(INDEX_PATH)
with open(INDEX_PATH + '.meta', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 加载embedding模型
model = SentenceTransformer(EMBEDDING_MODEL)

# 加载prompt模板
with open('prompts/base_prompt.txt', 'r', encoding='utf-8') as f:
    base_prompt = f.read()

# 设置OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')


def retrieve(query, top_k=TOP_K):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    docs = [corpus[i] for i in I[0]]
    return docs


def build_prompt(contexts, question):
    context_str = '\n'.join(contexts)
    prompt = base_prompt.replace('{context}', context_str).replace('{question}', question)
    return prompt


def ask_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    return response['choices'][0]['message']['content']


def main():
    print("欢迎使用RAG问答系统，输入你的问题（输入exit退出）：")
    while True:
        question = input("问题：")
        if question.strip().lower() == 'exit':
            break
        contexts = retrieve(question)
        prompt = build_prompt(contexts, question)
        answer = ask_gpt4(prompt)
        print("\n【答案】\n" + answer + "\n")

if __name__ == '__main__':
    main() 