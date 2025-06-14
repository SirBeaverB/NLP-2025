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
        self.topic_word_count = torch.zeros((num_topics, self.vocab_size), dtype=torch.int32, device=device)
        self.doc_topic_count = torch.zeros((self.num_docs, num_topics), dtype=torch.int32, device=device)
        self.topic_count = torch.zeros(num_topics, dtype=torch.int32, device=device)
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
        for it in trange(self.iterations, desc='LDA迭代'):
            for d_idx, doc in enumerate(self.docs):
                for w_idx, word in enumerate(doc):
                    old_topic = self.topic_assignments[d_idx][w_idx]
                    self.topic_word_count[old_topic, word] -= 1
                    self.doc_topic_count[d_idx, old_topic] -= 1
                    self.topic_count[old_topic] -= 1

                    # 采样新主题
                    p_z = (self.topic_word_count[:, word].float() + self.beta) / \
                          (self.topic_count.float() + self.vocab_size * self.beta) * \
                          (self.doc_topic_count[d_idx].float() + self.alpha)
                    p_z = p_z / p_z.sum()
                    # multinomial采样
                    new_topic = torch.multinomial(p_z, 1).item()

                    self.topic_assignments[d_idx][w_idx] = new_topic
                    self.topic_word_count[new_topic, word] += 1
                    self.doc_topic_count[d_idx, new_topic] += 1
                    self.topic_count[new_topic] += 1
            if (it+1) % 100 == 0 or it == 0:
                print(f"Iteration {it+1}/{self.iterations} finished.")
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
