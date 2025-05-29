import os
import json
import yaml
import faiss
import openai
from sentence_transformers import SentenceTransformer
import torch
import re


# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']
TOP_K = config['top_k']

# 加载embedding模型
model = SentenceTransformer(EMBEDDING_MODEL)
if torch.cuda.is_available():
    model = model.to('cuda')
    print('Using CUDA for embedding')
else:
    print('Using CPU for embedding')

# 加载faiss索引和元数据
if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
    res = faiss.StandardGpuResources()
    cpu_index = faiss.read_index(INDEX_PATH)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    print('Using FAISS GPU for search')
else:
    index = faiss.read_index(INDEX_PATH)
    print('Using FAISS CPU for search')
with open(INDEX_PATH + '.meta', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 加载prompt模板
with open('prompts/base_prompt.txt', 'r', encoding='utf-8') as f:
    base_prompt = f.read()

# 设置OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')


def simple_keyword_search(corpus, keywords, top_k=5):
    print("用于关键词检索的关键词:", keywords)
    scores = []
    for idx, doc in enumerate(corpus):
        content = doc.get('content', doc.get('text', ''))
        score = sum(content.count(k) for k in keywords)
        scores.append((score, idx))
    # 取分数最高的top_k
    scores.sort(reverse=True)
    top_idxs = [idx for score, idx in scores[:top_k] if score > 0]
    return [corpus[i] for i in top_idxs]


def retrieve(query, corpus, keywords, top_k=5):
    # embedding 检索
    query_emb = model.encode([query], device='cuda' if torch.cuda.is_available() else 'cpu')
    D, I = index.search(query_emb, top_k)
    emb_docs = [corpus[i] for i in I[0]]
    # 关键词检索
    kw_docs = simple_keyword_search(corpus, keywords, top_k=top_k)
    # 合并去重
    seen = set()
    merged = []
    for doc in kw_docs + emb_docs:
        doc_id = doc.get('url', '') + doc.get('title', '')
        if doc_id not in seen:
            merged.append(doc)
            seen.add(doc_id)
    return merged[:2*top_k]


def extract_relevant_snippet(content, question, keywords, max_length=1000):
    idx = -1
    for k in keywords:
        idx = content.find(k)
        if idx != -1:
            break
    if idx == -1:
        idx = 0
    start = max(0, idx - max_length // 2)
    end = min(len(content), start + max_length)
    return content[start:end]


def build_prompt(contexts, question, keywords):
    # 拼接每条的title、url、content等（只取相关片段）
    context_str = ''
    for doc in contexts:
        title = doc.get('title', '')
        url = doc.get('url', '')
        date = doc.get('date', '')
        content = doc.get('content', doc.get('text', ''))
        snippet = extract_relevant_snippet(content, question, keywords, max_length=1000)
        context_str += f"【标题】{title}\n【日期】{date}\n【链接】{url}\n【内容】{snippet}\n\n"
    prompt = base_prompt.replace('{context}', context_str).replace('{question}', question)
    return prompt


def ask_gpt4(prompt):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content


def extract_keywords_with_llm(question):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt = (
        "请从下面的问题中提取最关键的1个用于检索的关键词或短语，注意不要有过多定语，你想通过搜索这个关键词找到相关内容"
        "只输出用逗号分隔的关键词列表，不要解释。\n\n"
        "e.g. 问题：2024年我国文化和旅游部部长是谁？\n"
        "关键词：文化和旅游部部长\n"
        "e.g. 问题：2024年是中国红十字会成立多少周年？\n"
        "关键词：中国红十字会\n"
        f"问题：{question}"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64
    )
    keywords = response.choices[0].message.content.strip()
    # 处理成list
    return [k.strip() for k in keywords.split(',') if k.strip()]




def single_doc_agent(doc, question):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    title = doc.get('title', '')
    url = doc.get('url', '')
    date = doc.get('date', '')
    content = doc.get('content', doc.get('text', ''))
    snippet = extract_relevant_snippet(content, question, [], max_length=1000)
    context_str = f"【标题】{title}\n【日期】{date}\n【链接】{url}\n【内容】{snippet}\n"
    prompt = (
        "请判断下列语料内容是否已经足够唯一、准确地回答用户问题。如果足够，请直接给出简明、结构化的中文答案，并在最后一行输出"
        "参考链接：后面跟上该原文url；如果不足，请只回复：'信息不足'，不要编造内容。\n\n"
        f"【语料内容】\n{context_str}\n"
        f"【用户问题】\n{question}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def extract_evidence_agent(contexts, question, keywords, max_evidence=5):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    all_evidence = []
    
    # 对每篇文章单独调用agent
    for doc in contexts:
        title = doc.get('title', '')
        url = doc.get('url', '')
        date = doc.get('date', '')
        content = doc.get('content', doc.get('text', ''))
        
        context_str = f"【标题】{title}\n【日期】{date}\n【链接】{url}\n【内容】{content}\n"
        prompt = (
            f"请从下列语料内容中，筛选出可能可以回答用户问题的最相关的1-{max_evidence}条证据片段（可以是原文中的句子或段落），"
            "比如，用户问：澜湄六国是哪六个国家？那么出现在相关文章里的国家名都可能是证据，"
            "每条证据请用'证据X:'开头，后面跟上原文内容和对应的url。只输出证据，不要总结和解释。\n\n"
            f"【语料内容】\n{context_str}\n"
            f"【用户问题】\n{question}\n"
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )
        evidence = response.choices[0].message.content.strip()
        if evidence and evidence != '信息不足':
            all_evidence.append(evidence)
    
    # 合并所有证据
    return '\n\n'.join(all_evidence)


def summarize_agent(evidence_str, question):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt = (
        "请根据下列证据片段，综合推理并回答用户问题。"
        "请给出结构化、简明的中文答案，并在最后一行输出参考链接：后面列出所有用到的url（用逗号分隔，必须是证据中真实存在的url）；"
        "如果信息仍然不足，请只回复：'信息不足'，不要编造内容。\n\n"
        f"【证据片段】\n{evidence_str}\n"
        f"【用户问题】\n{question}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


def main():
    print("欢迎使用RAG问答系统，输入你的问题（输入exit退出）：")
    while True:
        question = input("问题：")
        if question.strip().lower() == 'exit':
            break
        # 先用 LLM agent 提取关键词
        keywords = extract_keywords_with_llm(question)
        print("LLM提取的关键词:", keywords)
        # 检索
        contexts = retrieve(question, corpus, keywords, top_k=5)
        # 先用单文档agent判断每一篇文档能否唯一回答
        found = False
        for doc in contexts:
            single_ans = single_doc_agent(doc, question)
            if single_ans != '信息不足':
                # 提取url
                match = re.search(r'参考链接[:：]\s*(\S+)', single_ans)
                url = match.group(1) if match else doc.get('url', '')
                ans = single_ans.split('\n')[0] if '参考链接' in single_ans else single_ans
                print(f"\n【答案】\n{ans}\n")
                if url:
                    print(f"【参考链接】\n{url}\n")
                found = True
                break
        if found:
            continue
        # 否则用证据抽取+推理agent
        evidence_str = extract_evidence_agent(contexts, question, keywords, max_evidence=5)
        print("\n【证据片段】\n" + evidence_str + "\n")
        summary_ans = summarize_agent(evidence_str, question)
        if summary_ans != '信息不足':
            match = re.search(r'参考链接[:：]\s*(\S+)', summary_ans)
            urls = []
            if match:
                urls = [u.strip() for u in match.group(1).split(',') if u.strip()]
            ans = summary_ans.split('\n')[0] if '参考链接' in summary_ans else summary_ans
            print(f"\n【答案】\n{ans}\n")
            if urls:
                print(f"【参考链接】")
                for url in urls:
                    print(url)
                print()
            continue
        # 否则走原有prompt
        prompt = build_prompt(contexts, question, keywords)
        answer = ask_gpt4(prompt)
        print("\n【答案】\n" + answer + "\n")

if __name__ == '__main__':
    main() 