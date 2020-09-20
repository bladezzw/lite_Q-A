# 句子向量 = 词向量的平均（出现在问句里的）， 如果给定的词没有出现在词典库里，则忽略掉这个词。
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle
from queue import PriorityQueue


f = open('data/model.pk', 'rb')
data_dict = pickle.load(f)
f.close()

inverted_idx = data_dict['inverted_idx']
qlist_seg = data_dict['qlist_seg']
alist = data_dict['alist']
X = data_dict['X']

# 加载nltk自带停用词，该停用词表个人感觉一般，具体到细分领域可能还是需要自己归纳
sw = set(stopwords.words('english'))
# 个人感觉对于一个问题而言这些词不应该删去
sw -= {'who', 'when', 'why', 'where', 'how'}
# 这里只是随便去了下符号
sw.update(['\'s', '``', '\'\''])
ps = PorterStemmer()


# 将GloVe转为word2vec
if not os.path.exists('./data/glove2word2vec.6B.100d.txt'):
    _ = glove2word2vec('./data/glove.6B.100d.txt', './data/glove2word2vec.6B.100d.txt')
    model = KeyedVectors.load_word2vec_format('./data/glove2word2vec.6B.100d.txt')
else:
    model = KeyedVectors.load_word2vec_format('./data/glove2word2vec.6B.100d.txt')


def text_preprocessing(text):
    """
    对单条文本进行处理。
    text: str类型

    return: 分词后的list
    """

    seg = list()
    # 直接用nltk分词
    for word in word_tokenize(text):
        # 小写化、词干提取
        word = ps.stem(word.lower())
        # 数值归一
        word = '#number' if word.isdigit() else word
        # 去停用词
        if len(word) > 1 and word not in sw:
            seg.append(word)

    return seg

def docvec_get(seg):
    """
    将分词数据转为句向量。
    seg: 分词后的数据

    return: 句向量
    """
    vector = np.zeros((1, 100))
    size = len(seg)
    for word in seg:
        try:
            vector += model.wv[word]
        except KeyError:
            size -= 1
    # 对向量求平均
    return vector / size





def top5results_emb(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q，转换成句子向量
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    # 用词向量后用词形还原更合理，此处就不做变更了
    seg = text_preprocessing(input_q)
    # 直接用上边建好的倒排表
    candidates = set()
    for word in seg:
        # 取所有包含任意一个词的文档的并集
        candidates = candidates | inverted_idx[word]
    candidates = list(candidates)

    q_vector = docvec_get(seg)
    # 计算问题向量的l2范数
    qnorm2 = np.linalg.norm(q_vector, axis=1, keepdims=True)
    q_vector = q_vector / qnorm2
    # 计算余弦相似度，前边已经l2规范化过，所以直接相乘
    sim = (X[candidates] @ q_vector.T)

    # 使用优先队列找出top5
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], candidates[cur]))
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    print([x[0] for x in pq_rank])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    return [alist[i] for i in top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案


if __name__ == '__main__':
    print(top5results_emb("Which airport was shut down?"))    # 在问题库中存在，经过对比，返回的首结果正确
    print(top5results_emb("Which airport is closed?"))
    print(top5results_emb("What government blocked aid after Cyclone Nargis?"))    # 在问题库中存在，经过对比，返回的首结果正确
    print(top5results_emb("Which government stopped aid after Hurricane Nargis?"))