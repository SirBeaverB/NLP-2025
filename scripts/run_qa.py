from query_kg_rag import extract_keywords_with_llm, answer_with_kg, retrieve, build_prompt, ask_qwen3, summarize_kg_rag, open_question_agent
import torch
import gc
import tqdm
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
#from transformers import pipeline

def evaluate(queries: list):
    """
    queries: List[str] 输入查询列表
    Return: List[str] 输出答案列表
    """
    # 初始化Qwen3-4B pipeline
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

    '''llm_model_name = "Qwen/Qwen3-4B"  # 或 "Qwen/Qwen3-4B"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype="auto",
        device_map="auto"
    )'''

    # 加载知识图谱
    kg_path = 'kg/kg.pkl'
    kg_utils = KGUtils(kg_path)

    answers = []
    from tqdm import tqdm
    for question in tqdm(queries, desc="Processing questions"):
        try:
            # 先用 LLM agent 提取关键词
            keywords, is_open = extract_keywords_with_llm(question)
            is_open = len(answers) >= 80  # 前80个是客观题，后20个是开放题
            
            if not is_open:
                try:
                    kg_answer, kg_entity = answer_with_kg(question, keywords)
                    
                    contexts = retrieve(question, corpus, keywords, top_k=20)
                    rag_answer = ''
                    rag_urls = []
                    
                    if contexts:
                        # 使用所有相关文章构建prompt
                        prompt = build_prompt(contexts, question, keywords)
                        rag_answer = ask_qwen3(prompt)
                        rag_urls = [doc.get('url', '') for doc in contexts if doc.get('url')]
                    
                    if rag_urls:
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
                    
                    fusion_answer = summarize_kg_rag(kg_answer, reread_answer if rag_urls and reread_contexts else rag_answer, question)
                    
                    if "信息不足" in fusion_answer or "未找到" in fusion_answer:
                        keywords, _ = extract_keywords_with_llm(question, max=2)
                        
                        kg_answer, kg_entity = answer_with_kg(question, keywords)
                        contexts = retrieve(question, corpus, keywords, top_k=20)
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
                    
                    answers.append(fusion_answer)
                except Exception as e:
                    print(f"处理客观题时出错: {str(e)}")
                    answers.append("出错")
            else:
                try:
                    kg_answer, kg_entity = answer_with_kg(question, keywords)
                    contexts = retrieve(question, corpus, keywords, top_k=20)
                    open_answer = open_question_agent(kg_answer, contexts, question, keywords)
                    answers.append(open_answer)
                except Exception as e:
                    print(f"处理开放题时出错: {str(e)}")
                    answers.append("出错")

            if is_open:
                # 提取答案中的链接
                urls = []
                if contexts:
                    urls = [doc.get('url', '') for doc in contexts if doc.get('url')]
                # 从答案中删除URL
                ans_text = answers[-1]
                for url in urls:
                    ans_text = ans_text.replace(url, '')
                answers[-1] = {
                    'ans': ans_text.strip(),
                    'reference': urls[:5]
                }

            # 清理CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                # 清理本轮变量
                del keywords, kg_answer, kg_entity, contexts, rag_answer, rag_urls
                gc.collect()
            except:
                pass
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
            answers.append("出错")
            
    return answers

if __name__ == '__main__':
     query = ["2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校?",
"2024年是中国红十字会成立多少周年",
"2024年我国文化和旅游部部长是谁?",
"《中华人民共和国爱国主义教育法》什么时候实施?",
"2023年全国电影总票房为多少元?",
"蒙古人民党总书记是谁?",
"2023-2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌?",
"福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新?",
"杭州第十九届亚洲运动会共举行多少天?",
"哪些单位在中国期刊高质量发展论坛的主论坛上做主题演讲?",
"绿水青山就是金山银山，请根据近期新闻，给我国的绿色发展建言献策",
"我国科技创新的重大成就有哪些?"]
query = ["2024年我国文化和旅游部部长是谁?","绿水青山就是金山银山，请根据近期新闻，给我国的绿色发展建言献策"]

import chardet

with open('queries.txt', 'rb') as f:
    raw = f.read()
    result = chardet.detect(raw)
    encoding = result['encoding']

with open('queries.txt', 'r', encoding=encoding) as f:
    query = [line.strip() for line in f if line.strip()]

ans = evaluate(query)
#print(ans)

# 保存答案到文件
with open('answers.json', 'w', encoding='utf-8') as f:
    json.dump(ans, f, ensure_ascii=False, indent=2)
print("answers saved in answers.json")
