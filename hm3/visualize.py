import matplotlib.pyplot as plt

def plot_top_words(phi, id2word, topic_id, top_k=10):
    """
    可视化某个主题下概率最高的top_k个词
    """
    top_word_ids = phi[topic_id].argsort()[::-1][:top_k]
    words = [id2word[i] for i in top_word_ids]
    probs = phi[topic_id][top_word_ids]
    plt.figure(figsize=(10, 5))
    plt.bar(words, probs)
    plt.title(f"Topic {topic_id} Top {top_k} Words")
    plt.ylabel("Probability")
    plt.xlabel("Word")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 