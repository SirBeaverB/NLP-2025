import pandas as pd
import networkx as nx
import os
import pickle
import yaml
import json
from collections import defaultdict
import glob

RELATION_MAP = {} # not needed

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_triples_from_folder(triples_folder):
    triples = []
    csv_files = glob.glob(os.path.join(triples_folder, "*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        triples.extend(df.values.tolist())
    return triples

def load_entities(entities_path):
    if not entities_path or not os.path.exists(entities_path):
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
    triples_folder = config['kg_triples_folder']  # 例如 'kg/triples'
    entities = load_entities(config.get('kg_entities_path', ''))
    triples = load_triples_from_folder(triples_folder)
    G = build_kg(triples, entities)
    print(f"实体数: {G.number_of_nodes()}，三元组数: {G.number_of_edges()}")

    save_kg(G, config['kg_index_path'])
    print(f"KG saved to {config['kg_index_path']}")

if __name__ == "__main__":
    main() 