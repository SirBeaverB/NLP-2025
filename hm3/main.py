import os
import numpy as np
from utils import clean_text, tokenize
from lda import LDA
from visualize import plot_top_words
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
try:
    # 检查是否有中文字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in font_list if any(c in f for c in ['Chinese', 'CJK', 'WenQuanYi', 'SimHei'])]
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
    print(f"可用的中文字体: {chinese_fonts[:5]}")
except:
    pass


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
lda = LDA(docs, vocab, num_topics=10, alpha=0.1, beta=0.01, iterations=500)
lda.run()
lda.save_params('theta.npy', 'phi.npy')



# 保存作图所需的数据文件
lda.save_params('theta.npy', 'phi.npy')
# 保存词表映射
np.save('word2id.npy', lda.word2id)
np.save('id2word.npy', lda.id2word)

# 读取保存的数据文件进行可视化
phi = np.load('phi.npy')
word2id = np.load('word2id.npy', allow_pickle=True).item()
id2word = np.load('id2word.npy', allow_pickle=True).item()

# 可视化每个主题的Top10词并保存图片
for topic_id in range(10):
    plt.figure(figsize=(10, 5))
    plot_top_words(phi, id2word, topic_id, top_k=10)
    plt.savefig(f'topic_{topic_id}_top_words.png')
    plt.close()
    # 将所有主题词保存到一个文件
    with open('all_topics_words.txt', 'w', encoding='utf-8') as f:
        for topic_id in range(10):
            f.write(f"Topic {topic_id} Top 10 Words:\n")
            f.write("=" * 30 + "\n")
            # 获取主题下概率最高的10个词
            top_word_ids = phi[topic_id].argsort()[::-1][:10]
            for i, word_id in enumerate(top_word_ids, 1):
                word = id2word[word_id]
                prob = phi[topic_id][word_id]
                f.write(f"{i:2d}. {word:<15} (prob: {prob:.4f})\n")
            f.write("\n")

