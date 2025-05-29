import networkx as nx
import difflib
import pickle
import json

class KGUtils:
    def __init__(self, kg_index_path, entities_path=None):
        with open(kg_index_path, 'rb') as f:
            self.G = pickle.load(f)
        self.alias_map = {}
        if entities_path and entities_path.endswith('.json'):
            try:
                with open(entities_path, 'r', encoding='utf-8') as f:
                    entities = json.load(f)
                for ent, attrs in entities.items():
                    self.alias_map[ent] = ent
                    for alias in attrs.get('aliases', []):
                        self.alias_map[alias] = ent
            except Exception:
                pass

    def find_entity(self, name, threshold=0.7):
        """先查别名映射，再模糊匹配实体名"""
        name = name.strip().replace('　', '').replace(' ', '')
        if name in self.alias_map:
            return self.alias_map[name]
        entities = list(self.G.nodes)
        matches = difflib.get_close_matches(name, entities, n=1, cutoff=threshold)
        return matches[0] if matches else None

    def get_triples(self, entity):
        """获取与实体相关的三元组"""
        triples = []
        for nbr in self.G.neighbors(entity):
            rel = self.G[entity][nbr]['relation']
            triples.append((entity, rel, nbr))
        for nbr in self.G.predecessors(entity):
            rel = self.G[nbr][entity]['relation']
            triples.append((nbr, rel, entity))
        return triples

    def shortest_path(self, source, target):
        """实体间最短路径"""
        try:
            path = nx.shortest_path(self.G, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None 