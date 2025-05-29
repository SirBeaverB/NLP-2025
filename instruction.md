本项目旨在构建一个通用的基于 RAG（Retrieval-Augmented Generation）架构的问答系统。该系统可以回答用户提出的任意问题，并基于指定语料库（人民日报）中的内容进行检索和生成。

用户输入问题后，系统会：

1. 从语料库中检索相关内容片段

2. 将检索结果作为上下文，输入语言模型

3. 输出结构化、可信的自然语言答案


## 爬虫
已完成。

## agent构建

构建一个rag，使用langchain架构，构建一个知识图谱

用户问题 → 语义检索模块 → Top-K 文档 → 拼接上下文 → LLM Prompt → 回答

``` 
rag-qa-system/
├── data/                         
│   └── corpus/                   # 原始语料
│       ├── 2023-05-01.json
│       └── ...
│── kg/                       # 知识图谱文件（结构化）
│   ├── triples.csv           # 三元组：head, relation, tail
│   ├── entities.json         # 可选：实体定义、描述
│   └── schema.yaml           # 可选：知识图谱结构定义
├── index/                        
│   ├── faiss_index.bin           # RAG索引
│   └── kg_index.pkl              # KG的实体或图索引缓存
├── scripts/
│   ├── build_index.py            # 语料编码 + FAISS 索引
│   ├── build_kg.py               # 构建 KG 索引/图对象
│   ├── extract_triples_ltp.py    # 放弃
│   ├── extract_triples_mrebel.py # 构建三元组
│   ├── li.py                     # 放弃
│   ├── query_rag.py              # 普通 RAG 问答
│   ├── query_kg_rag.py           # KG + RAG 融合问答
│   └── utils_kg.py               # KG 操作工具类（路径查找、实体匹配等）
├── prompts/
│   ├── base_prompt.txt           # 普通 RAG Prompt
│   └── kg_prompt.txt             # KG + RAG 融合 Prompt
├── requirements.txt              
└── config.yaml                   # 参数配置，如 top_k、使用 KG 与否等

```
