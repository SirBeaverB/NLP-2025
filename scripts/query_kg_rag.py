import os
import json
import yaml
import faiss
from sentence_transformers import SentenceTransformer
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from utils_kg import KGUtils
import gc
from torch.nn.parallel import DataParallel

'''
openai的API有点贵，还容易超token，只用做关键词提取
直接改用Qwen2.5-7B本地部署
'''

# Load model directly
#from transformers import pipeline

# 初始化Qwen2.5-7B pipeline
#qwen_pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-instruct")

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INDEX_PATH = config['index_path']
EMBEDDING_MODEL = config['embedding_model']
TOP_K = config['top_k']

# 加载embedding模型
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

def encode_query(texts, device, batch_size=32):
    import torch
    if isinstance(model, DataParallel):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = model.module.encode(batch, device=device, convert_to_tensor=True)
            emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
            all_emb.append(emb)
        return torch.cat(all_emb, dim=0).cpu().numpy()
    else:
        emb = model.encode(texts, device=device, convert_to_tensor=True)
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        return emb.cpu().numpy()


# 加载faiss索引和元数据
if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
    ngpus = faiss.get_num_gpus()
    print(f"Using {ngpus} GPUs for FAISS search")
    cpu_index = faiss.read_index(INDEX_PATH)
    index = faiss.index_cpu_to_all_gpus(cpu_index)
else:
    index = faiss.read_index(INDEX_PATH)
    print('Using FAISS CPU for search')
with open(INDEX_PATH + '.meta', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 加载prompt模板
with open('prompts/base_prompt.txt', 'r', encoding='utf-8') as f:
    base_prompt = f.read()

llm_model_name = "Qwen/Qwen2.5-7B-instruct"  # 或 "Qwen/Qwen3-4B"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 加载知识图谱
kg_path = 'kg/kg.pkl'
kg_utils = KGUtils(kg_path)

def qwen_chat(messages, max_new_tokens=2048):
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    content = llm_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

def simple_keyword_search(corpus, keywords, top_k=20):
    print("用于关键词检索的关键词:", keywords)
    scores = []
    for idx, doc in enumerate(corpus):
        content = doc.get('chunk_content', doc.get('text', ''))
        #title = doc.get('title', '')
        full_text = content or ''
        score = sum(full_text.count(k) for k in keywords)
        scores.append((score, idx))
    # 取分数最高的top_k
    scores.sort(reverse=True)
    top_idxs = [idx for score, idx in scores[:top_k] if score > 0]
    return [corpus[i] for i in top_idxs]



def retrieve(query, corpus, keywords, top_k=20):
    # embedding 检索
    query_emb = encode_query([query], device=device)
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


def expand_chunk_context(chunk, all_chunks, window=1):
    """
    给定一个chunk，返回它前后window个chunk合并的内容
    """
    article_id = chunk.get('article_id') or chunk.get('source') or chunk.get('filename')
    chunk_id = chunk['chunk_id']
    article_chunks = [c for c in all_chunks if (c.get('article_id') or c.get('source') or c.get('filename')) == article_id]
    article_chunks = sorted(article_chunks, key=lambda x: x['chunk_id'])
    idx = next((i for i, c in enumerate(article_chunks) if c['chunk_id'] == chunk_id), None)
    if idx is None:
        return chunk.get('chunk_content', chunk.get('content', chunk.get('text', '')))
    start = max(0, idx - window)
    end = min(len(article_chunks), idx + window + 1)
    expanded = ''.join([c.get('chunk_content', c.get('content', c.get('text', ''))) for c in article_chunks[start:end]])
    return expanded


def build_prompt(contexts, question, keywords):
    context_str = ''
    for doc in contexts:
        title = doc.get('title', '')
        url = doc.get('url', '')
        date = doc.get('date', '')
        content = doc.get('expanded_content', doc.get('chunk_content', doc.get('content', doc.get('text', ''))))
        snippet = extract_relevant_snippet(content, question, keywords, max_length=1000)
        context_str += f"【标题】{title}\n【日期】{date}\n【链接】{url}\n【内容】{snippet}\n\n"
    prompt = base_prompt.replace('{context}', context_str).replace('{question}', question)
    return prompt


def get_qwen_generated_text(result):
    gen = result[0]['generated_text']
    # 兼容字符串、列表（字符串/字典）等多种情况
    if isinstance(gen, str):
        return gen.strip()
    elif isinstance(gen, list):
        # 只拼接字符串部分
        return ''.join(
            [item['generated_text'] if isinstance(item, dict) and 'generated_text' in item else str(item)
             for item in gen]
        ).strip()
    elif isinstance(gen, dict) and 'content' in gen:
        return gen['content'].strip()
    else:
        return str(gen).strip()


def ask_qwen3(prompt):
    messages = [{"role": "user", "content": prompt}]
    return qwen_chat(messages)


def extract_keywords_with_llm(question, max=1):
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt = (
        f"请从下面的问题中提取最关键的{max}个用于检索的关键词或短语，能少就少，注意不要有过多定语，但是也不要拆开专有名词，比如用书名号、引号括起来的书名、会议名等，你想通过搜索这个关键词找到相关内容。"
        "eg 获得第二十七届中国青年五四奖章的女性有谁？ 关键词：中国青年五四奖章"
        "eg 《习近平新时代中国特色社会主义思想专题摘编》民族文字版共有几个出版社参与发行？ 关键词：《习近平新时代中国特色社会主义思想专题摘编》"
        "此外，请判断该问题是否为开放题（即答案不是唯一事实、需要主观判断或综合分析），如果是开放题请输出True，否则输出False。"
        "输出格式严格如下：\n关键词：xxx,yyy\n开放题：True/False\n"
        "e.g. 问题：2024年我国文化和旅游部部长是谁？\n关键词：文化和旅游部部长\n开放题：False\n"
        "e.g. 问题：你如何看待人工智能对未来社会的影响？\n关键词：人工智能,未来社会\n开放题：True\n"
        f"问题：{question}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=128
    )
    content = response.choices[0].message.content.strip()
    # 解析关键词和开放题bool
    keywords = []
    is_open = False
    for line in content.splitlines():
        if line.startswith('关键词：'):
            keywords = [k.strip() for k in line.replace('关键词：','').split(',') if k.strip()]
        if line.startswith('开放题：'):
            is_open = 'true' in line.lower()
    return keywords, is_open



# 知识图谱问答函数

def answer_with_kg(question, keywords):
    # 只用第一个关键词做实体查找
    if not keywords:
        return '未提取到关键词，无法用KG回答', None
    entity = kg_utils.find_entity(keywords[0])
    if not entity:
        return f'知识图谱中未找到实体：{keywords[0]}', None
    triples = kg_utils.get_triples(entity)
    if not triples:
        return f'知识图谱中未找到与实体"{entity}"相关的三元组', entity
    # 简单拼接三元组作为答案
    triple_strs = [f'{h} --{r}--> {t}' for h, r, t in triples]
    answer = f'实体"{entity}"相关的知识：\n' + '\n'.join(triple_strs)
    return answer, entity

# 综合KG和RAG答案
def summarize_kg_rag(kg_answer, rag_answer, question):
    prompt = (
        "请根据以下两部分信息，综合推理并用中文简明地回答用户问题，不需做出解释，不要重复问题，只给出答案即可。\n"
        "【KG答案】：\n" + kg_answer + "\n"
        "【RAG答案】：\n" + rag_answer + "\n"
        "【用户问题】：\n" + question + "\n"
        "请综合两部分信息，优先使用更rag所给出的更权威、直接的内容，必要时可融合推理。若信息不足请直接回复信息不足。"
    )
    messages = [{"role": "user", "content": prompt}]
    return qwen_chat(messages)

def open_question_agent(kg_answer, rag_docs, question, keywords):
    # 拼接KG和RAG内容
    rag_context = ''
    for doc in rag_docs:
        title = doc.get('title', '')
        url = doc.get('url', '')
        date = doc.get('date', '')
        content = doc.get('content', doc.get('text', ''))
        snippet = extract_relevant_snippet(content, question, keywords, max_length=1000)
        rag_context += f"【标题】{title}\n【日期】{date}\n【链接】{url}\n【内容】{snippet}\n\n"
    prompt = (
        "你是一个知识型问答助手。请综合下列知识图谱信息和检索到的多篇文章内容，给出一个自由度较高、结构化、条理清晰的中文回答。"
        "可以适当分析、归纳、推理，允许有主观判断，但要基于提供的内容。不多于512个字, 不要有多于5个url。"
        "如有引用内容，请在段末注明来源url。若信息不足可适当发挥，但不要编造具体事实。\n\n"
        f"【知识图谱信息】\n{kg_answer}\n\n"
        f"【相关文章内容】\n{rag_context}\n"
        f"【用户问题】\n{question}\n"
    )
    return ask_qwen3(prompt)

def main():
    #print("欢迎使用RAG+KG问答系统，输入你的问题（输入exit退出）：")
    while True:
        question = input("问题：")
        if question.strip().lower() == 'exit':
            break
        # 先用 LLM agent 提取关键词
        keywords, is_open = extract_keywords_with_llm(question)
        #print("LLM提取的关键词:", keywords)
        #print("是否为开放题:", is_open)
        if not is_open:
            kg_answer, kg_entity = answer_with_kg(question, keywords)
            print("\n【KG答案】\n" + kg_answer + "\n")
            # RAG检索最相关一篇
            contexts = retrieve(question, corpus, keywords, top_k=20)
            for doc in contexts:
                doc['expanded_content'] = expand_chunk_context(doc, corpus, window=1)
            rag_answer = ''
            rag_urls = []
            if contexts:
                # 使用所有相关文章构建prompt
                prompt = build_prompt(contexts, question, keywords)
                rag_answer = ask_qwen3(prompt)
                rag_urls = [doc.get('url', '') for doc in contexts if doc.get('url')]
            #print("\n【RAG答案】\n" + rag_answer + "\n")
            #print("\n【汇总】")
            #print("KG答案：" + kg_answer)
            #print("RAG答案：" + rag_answer)
            if rag_urls:
                #print("RAG参考链接：" + ', '.join(rag_urls))
                # reread: 重新读取参考链接对应的文章内容
                reread_contexts = []
                for url in rag_urls:
                    for doc in corpus:
                        if doc.get('url', '') == url:
                            reread_contexts.append(doc)
                            break
                if reread_contexts:
                    reread_prompt = build_prompt(reread_contexts, question, keywords)
                    reread_answer = ask_qwen3(reread_prompt)
                    print("\n【Reread最终答案】\n" + reread_answer + "\n")
            fusion_answer = summarize_kg_rag(kg_answer, reread_answer if rag_urls and reread_contexts else rag_answer, question)
            print(fusion_answer)
            
            if "信息不足" in fusion_answer or "未找到" in fusion_answer:
                # 重新提取更多关键词
                keywords, _ = extract_keywords_with_llm(question, max=2)
                
                # 重新进行KG和RAG检索
                kg_answer, kg_entity = answer_with_kg(question, keywords)
                contexts = retrieve(question, corpus, keywords, top_k=20)
                for doc in contexts:
                    doc['expanded_content'] = expand_chunk_context(doc, corpus, window=1)
                rag_answer = ''
                rag_urls = []
                
                if contexts:
                    prompt = build_prompt(contexts, question, keywords)
                    rag_answer = ask_qwen3(prompt)
                    rag_urls = [doc.get('url', '') for doc in contexts if doc.get('url')]
                
                if rag_urls:
                    reread_contexts = []
                    for url in rag_urls:
                        for doc in corpus:
                            if doc.get('url', '') == url:
                                reread_contexts.append(doc)
                                break
                    if reread_contexts:
                        reread_prompt = build_prompt(reread_contexts, question, keywords)
                        reread_answer = ask_qwen3(reread_prompt)
                
                fusion_answer = summarize_kg_rag(kg_answer, reread_answer if rag_urls and reread_contexts else rag_answer, question)
                print("\n【重试后的答案】\n" + fusion_answer)
            
        else:
            # 改成搜索文章切块之后，需要搜索更多chunk，但最后只给出最多5个url
            kg_answer, kg_entity = answer_with_kg(question, keywords)
            #print("\n【KG答案】\n" + kg_answer + "\n")
            contexts = retrieve(question, corpus, keywords, top_k=20)
            for doc in contexts:
                doc['expanded_content'] = expand_chunk_context(doc, corpus, window=1)
            #print("\n【RAG相关文章数量】", len(contexts))
            # 综合生成开放性回答
            open_answer = open_question_agent(kg_answer, contexts, question, keywords)
            print(open_answer)
            #print("\n-----------------------------\n")

        # 清理CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            # 清理本轮变量
            del keywords, kg_answer, kg_entity, contexts, rag_answer, rag_urls
            gc.collect()
        except:
            pass


if __name__ == '__main__':
    main() 