import os
import yaml
import pickle
from scripts.utils_kg import KGUtils
from scripts.query_rag import retrieve_rag_context, call_llm


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_kg_prompt(kg_triples, rag_context, question, prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    triples_str = "\n".join([f"{h} -[{r}]-> {t}" for h, r, t in kg_triples])
    rag_context_str = "\n".join([doc['content'] for doc in rag_context])
    return prompt_template.format(kg_triples=triples_str, rag_context=rag_context_str, question=question)

def extract_entities_from_question(question, kg_utils):
    # 优先匹配长别名和实体名
    found = set()
    for ent in sorted(kg_utils.alias_map.keys(), key=lambda x: -len(x)):
        if ent in question:
            found.add(kg_utils.alias_map[ent])
    return list(found)

def main():
    config = load_config()
    kg_utils = KGUtils(config['kg_index_path'], config.get('kg_entities_path', None))
    question = input("请输入你的问题：")
    # 改进实体抽取：优先匹配长别名和实体名
    entities = extract_entities_from_question(question, kg_utils)
    kg_triples = []
    for ent in entities:
        kg_triples.extend(kg_utils.get_triples(ent))
    # RAG 检索
    rag_context = retrieve_rag_context(question, config)
    # 构建Prompt
    prompt = build_kg_prompt(kg_triples, rag_context, question, 'prompts/kg_prompt.txt')
    # 调用LLM
    answer = call_llm(prompt, config)
    print("\n答案：\n", answer)
    print("\n参考三元组：")
    for h, r, t in kg_triples:
        print(f"{h} -[{r}]-> {t}")

if __name__ == "__main__":
    main() 