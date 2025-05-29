import pandas as pd
import networkx as nx
import os
import pickle
import yaml
import json
from collections import defaultdict

RELATION_MAP = {
    # 身份/职务
    "为": "担任",
    "是": "担任",
    "现任": "担任",
    "曾任": "担任",
    "任职": "担任",
    "任": "担任",
    "担任过": "担任",
    # 所属/归属
    "属于": "属于",
    "隶属于": "属于",
    "是...的": "属于",
    # 地点/位置
    "位于": "位于",
    # 教育/毕业
    "毕业于": "毕业院校",
    "毕业自": "毕业院校",
    "就读于": "毕业院校",
    # 工作单位
    "工作于": "工作单位",
    "服务于": "工作单位",
    "就职于": "工作单位",
    # 获得/荣誉
    "获得": "获得",
    "获得了": "获得",
    "获得过": "获得",
    "获得奖项": "获得",
    "获得荣誉": "获得",
    "获得学位": "获得",
    "获得称号": "获得",
    "获得资格": "获得",
    "获得证书": "获得",
    "获得专利": "获得",
    # 创立/主办
    "创立": "创立",
    "创建": "创立",
    "成立": "创立",
    "建立": "创立",
    "创办": "创立",
    "发起": "创立",
    "主办": "主办",
    "举办": "主办",
    # 发表/出版
    "出版": "出版",
    "发表": "发表",
    # 编辑
    "编辑": "编辑",
    # 出生地
    "出生于": "出生地",
    "出生地": "出生地"
}

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_triples(triples_path):
    df = pd.read_csv(triples_path)
    triples = df.values.tolist()
    return triples

def load_entities(entities_path):
    if not os.path.exists(entities_path):
        return {}
    with open(entities_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_entity(name, alias_map):
    # 统一全角半角、去空格
    name = str(name).strip().replace('　', '').replace(' ', '')
    # 别名映射
    return alias_map.get(name, name)

def build_alias_map(entities):
    alias_map = {}
    for ent, attrs in entities.items():
        alias_map[ent] = ent
        for alias in attrs.get('aliases', []):
            alias_map[alias] = ent
    return alias_map

def build_kg(triples, entities=None):
    G = nx.DiGraph()
    alias_map = build_alias_map(entities or {})
    seen = set()
    for h, r, t in triples:
        # 关系归一化
        r = RELATION_MAP.get(r, r)
        # 实体标准化
        h_norm = normalize_entity(h, alias_map)
        t_norm = normalize_entity(t, alias_map)
        # 去重和无效三元组过滤
        if h_norm == t_norm:
            continue
        triple_key = (h_norm, r, t_norm)
        if triple_key in seen:
            continue
        seen.add(triple_key)
        G.add_edge(h_norm, t_norm, relation=r)
    # 补全实体属性
    if entities:
        for ent, attrs in entities.items():
            if ent in G:
                G.nodes[ent].update(attrs)
    return G

def save_kg(G, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(G, f)

def main():
    config = load_config()
    triples = load_triples(config['kg_triples_path'])
    entities = load_entities(config.get('kg_entities_path', ''))
    G = build_kg(triples, entities)
    print(f"实体数: {G.number_of_nodes()}，三元组数: {G.number_of_edges()}")
    save_kg(G, config['kg_index_path'])
    print(f"KG saved to {config['kg_index_path']}")

if __name__ == "__main__":
    main() 