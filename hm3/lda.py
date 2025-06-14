import torch
import random
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LDA:
    def __init__(self, docs, vocab, num_topics=10, alpha=0.1, beta=0.01, iterations=1000):
        self.docs = docs  # list of list of word ids
        self.vocab = vocab  # list of words
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.num_docs = len(docs)
        self.vocab_size = len(vocab)
        self.topic_word_count = torch.zeros((num_topics, self.vocab_size), dtype=torch.float32, device=device)
        self.doc_topic_count = torch.zeros((self.num_docs, num_topics), dtype=torch.float32, device=device)
        self.topic_count = torch.zeros(num_topics, dtype=torch.float32, device=device)
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in enumerate(vocab)}
        self.topic_assignments = []
        self.theta = None  # 文档-主题分布
        self.phi = None    # 主题-词分布
        self._initialize()

    def _initialize(self):
        # 随机初始化主题分配
        for d_idx, doc in enumerate(self.docs):
            current_doc_topics = []
            for word in doc:
                topic = random.randint(0, self.num_topics - 1)
                current_doc_topics.append(topic)
                self.topic_word_count[topic, word] += 1
                self.doc_topic_count[d_idx, topic] += 1
                self.topic_count[topic] += 1
            self.topic_assignments.append(current_doc_topics)

    def run(self):
        from tqdm import trange
        beta_sum = self.vocab_size * self.beta
        alpha_sum = self.num_topics * self.alpha

        # 所有张量移至 GPU
        self.topic_word_count = self.topic_word_count.to(device)  # [K, V]
        self.doc_topic_count = self.doc_topic_count.to(device)    # [D, K]
        self.topic_count = self.topic_count.to(device)            # [K]
        self.topic_assignments = [torch.tensor(t, device=device) for t in self.topic_assignments]

        # 文档也转为张量
        doc_tensors = [torch.tensor(doc, device=device) for doc in self.docs]

        # 增加批处理大小
        batch_size = 128
        num_docs = len(self.docs)

        for it in trange(self.iterations, desc='LDA迭代'):
            for batch_start in range(0, num_docs, batch_size):
                batch_end = min(batch_start + batch_size, num_docs)
                batch_docs = doc_tensors[batch_start:batch_end]
                batch_assignments = self.topic_assignments[batch_start:batch_end]
                batch_doc_topic = self.doc_topic_count[batch_start:batch_end]

                for local_idx, (doc_words, topics) in enumerate(zip(batch_docs, batch_assignments)):
                    d_idx = batch_start + local_idx
                    doc_len = len(doc_words)

                    # 对每个词采样新 topic（向量化）
                    word_ids = doc_words
                    old_topics = topics

                    # --- remove old assignments ---
                    for topic in range(self.num_topics):
                        mask = (old_topics == topic)
                        if mask.any():
                            self.topic_word_count[topic].scatter_add_(0, word_ids[mask], -torch.ones_like(word_ids[mask], dtype=torch.float32))
                            batch_doc_topic[local_idx, topic] -= mask.sum()
                            self.topic_count[topic] -= mask.sum()

                    # --- compute conditional probability p(z|w,d) ---
                    topic_word_probs = (self.topic_word_count[:, word_ids] + self.beta) / \
                                    (self.topic_count[:, None] + beta_sum)        # shape: [K, doc_len]
                    doc_topic_probs = (batch_doc_topic[local_idx] + self.alpha).unsqueeze(1) / \
                                    (doc_len - 1 + alpha_sum)
                    probs = topic_word_probs * doc_topic_probs                     # shape: [K, doc_len]
                    
                    # 使用log-sum-exp技巧提高数值稳定性
                    log_probs = torch.log(probs + 1e-10)
                    log_probs = log_probs - log_probs.max(dim=0, keepdim=True)[0]
                    probs = torch.exp(log_probs)
                    probs = probs / probs.sum(dim=0, keepdim=True)

                    # --- sample new topics ---
                    new_topics = torch.multinomial(probs.t(), 1).squeeze()        # shape: [doc_len]

                    # --- update assignments and counts ---
                    topics[:] = new_topics
                    for topic in range(self.num_topics):
                        mask = (new_topics == topic)
                        if mask.any():
                            self.topic_word_count[topic].scatter_add_(0, word_ids[mask], torch.ones_like(word_ids[mask], dtype=torch.float32))
                            batch_doc_topic[local_idx, topic] += mask.sum()
                            self.topic_count[topic] += mask.sum()

                # 每处理4个批次清理一次GPU缓存
                if (batch_end - batch_start) % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()

            # 打印统计信息
            if (it + 1) % 50 == 0 or it == 0:
                print(f"Iteration {it+1}/{self.iterations} finished.")
                topic_stats = self.topic_count.float().cpu().numpy()
                print(f"Topic distribution stats:")
                print(f"  Mean: {topic_stats.mean():.2f}")
                print(f"  Std: {topic_stats.std():.2f}")
                print(f"  Min: {topic_stats.min():.2f}")
                print(f"  Max: {topic_stats.max():.2f}")
                print(f"  Top 3 topics: {topic_stats.argsort()[-3:][::-1]}")

        # 还原到 CPU
        self.topic_assignments = [ta.cpu().tolist() for ta in self.topic_assignments]
        self._compute_theta_phi()



    def _compute_theta_phi(self):
        # 文档-主题分布
        self.theta = (self.doc_topic_count.float() + self.alpha)
        self.theta = self.theta / self.theta.sum(dim=1, keepdim=True)
        # 主题-词分布
        self.phi = (self.topic_word_count.float() + self.beta)
        self.phi = self.phi / self.phi.sum(dim=1, keepdim=True)

    def save_params(self, theta_path, phi_path):
        # 保存时转为cpu numpy数组
        import numpy as np
        np.save(theta_path, self.theta.cpu().numpy())
        np.save(phi_path, self.phi.cpu().numpy())

    def printTopK(self, topic_id, K=10):
        phi_cpu = self.phi.cpu().numpy()
        import numpy as np
        top_word_ids = np.argsort(phi_cpu[topic_id])[::-1][:K]
        print(f"Topic {topic_id} Top {K} Words:")
        for idx in top_word_ids:
            print(self.id2word[idx], end=' ')
        print()