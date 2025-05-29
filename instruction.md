本项目旨在构建一个通用的基于 RAG（Retrieval-Augmented Generation）架构的问答系统。该系统可以回答用户提出的任意问题，并基于指定语料库（人民日报）中的内容进行检索和生成。

用户输入问题后，系统会：

1. 从语料库中检索相关内容片段

2. 将检索结果作为上下文，输入语言模型

3. 输出结构化、可信的自然语言答案


## 爬虫
已完成。

## agent构建

构建一个rag，使用langchain架构

用户问题 → 语义检索模块 → Top-K 文档 → 拼接上下文 → LLM Prompt → 回答

``` 
rag-qa-system/
├── data/                         # 语料文件
│   ├── 2023-05-01.json
│   ├── 2023-05-02.json
│   └── ...
├── index/                        # 检索索引
│   └── faiss_index.bin
├── scripts/
│   ├── build_index.py           # 构建语料索引
│   └── query_rag.py             # 输入问题，生成回答
├── prompts/
│   └── base_prompt.txt          # Prompt 模板
├── requirements.txt             # 所需依赖库
└── config.yaml                  # 参数配置，如top_k、模型类型等
```
