import os
import numpy as np
from utils import clean_text, tokenize
from lda import LDA
from visualize import plot_top_words
import matplotlib.pyplot as plt


# 数据目录
DATA_DIR = 'sampled_dataset'
CATEGORIES = os.listdir(DATA_DIR)

# 读取所有文本
texts = []
for cat in CATEGORIES:
    cat_dir = os.path.join(DATA_DIR, cat)
    if not os.path.isdir(cat_dir):
        continue
    for fname in os.listdir(cat_dir):
        fpath = os.path.join(cat_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())

# 文本清洗和分词
doc_tokens = []
for text in texts:
    clean = clean_text(text)
    tokens = tokenize(clean)
    if tokens:
        doc_tokens.append(tokens)

# 构建词表和文档-词id表示
vocab = list(set([w for doc in doc_tokens for w in doc]))
word2id = {w: i for i, w in enumerate(vocab)}
docs = [[word2id[w] for w in doc if w in word2id] for doc in doc_tokens]

# LDA建模
lda = LDA(docs, vocab, num_topics=10, alpha=0.5, beta=0.1, iterations=300)
lda.run()
lda.save_params('theta.npy', 'phi.npy')

# 可视化每个主题的Top10词并保存图片
for topic_id in range(10):
    plt.figure(figsize=(10, 5))
    plot_top_words(lda.phi, lda.id2word, topic_id, top_k=10)
    plt.savefig(f'topic_{topic_id}_top_words.png')
    plt.close()
    lda.printTopK(topic_id, K=10)
