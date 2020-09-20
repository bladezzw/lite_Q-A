# TODO: 对于qlist做文本预处理操作（不要对alist进行操作，因为答案直接调用即可，我们只匹配问题）。 可以考虑以下几种操作：
#       1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
#       2. 转换成lower_case： 这是一个基本的操作
#       3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
#       4. 去掉出现频率很低的词：比如出现次数少于10,20....
#       5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
#       6. stemming（利用porter stemming): 因为是英文，所以stemming也是可以做的工作
#       7. 也可以做下词形还原，但是在这个任务里个人感觉不是很有必要
#       请注意，不一定要按照上面的顺序来处理
#       齐夫定律（英语：Zipf's law，IPA/ˈzɪf/）是由哈佛大学的语言学家乔治·金斯利·齐夫（George Kingsley Zipf）于1949年发表的实验定律。它可以表述为：在自然语言的语料库里，一个单词出现的频率与它在频率表里的排名成反比。所以，频率最高的单词出现的频率大约是出现频率第二位的单词的2倍，而出现频率第二位的单词则是出现频率第四位的单词的2倍。这个定律被作为任何与幂定律概率分布有关的事物的参考。 [1]
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from data_process import read_corpus

qlist, alist = read_corpus('./data/train-v2.0.json')



# 加载nltk自带停用词，该停用词表个人感觉一般，具体到细分领域可能还是需要自己归纳
sw = set(stopwords.words('english'))
# 个人感觉对于一个问题而言这些词不应该删去
sw -= {'who', 'when', 'why', 'where', 'how'}
# 这里只是随便去了下符号
sw.update(['\'s', '``', '\'\''])
ps = PorterStemmer()


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


words_cnt = Counter()
qlist_seg = list()
for text in qlist:
    seg = text_preprocessing(text)
    qlist_seg.append(seg)
    words_cnt.update(seg)

value_sort = sorted(words_cnt.values(), reverse=True)
# 根据Zipf定律计算99%覆盖率下的过滤词频，解释见程序下边
min_tf = value_sort[int(math.exp(0.99 * math.log(len(words_cnt))))]
for cur in range(len(qlist_seg)):
    qlist_seg[cur] = [word for word in qlist_seg[cur] if words_cnt[word] > min_tf]

print(qlist_seg[:10])
# Zipf's law一个实验定律，按照从最常见到非常见排列，第二常见的频率是最常见频率的出现次数的1/2，第三常见的频率是最常见的频率的1/3，第n常见的频率是最常见频率出现次数的1/n。
#
# 假设我们文本的词频符合该定律，那么对1/n进行积分得到ln(n)，为了使99%的文本得到覆盖则需ln(x)>0.99*ln(n)，n是词type数，x是词频从高到底排列时的阈值分割点，最后x=e^(0.99*ln(n))。


# 文本表示

# 做完关键的预处理过程之后，就需要把每一个文本转换成向量。
# TODO: 把qlist中的每一个问题字符串转换成tf-idf向量, 转换之后的结果存储在X矩阵里。 X的大小是： N* D的矩阵。 这里N是问题的个数（样本个数），
#       D是字典库的大小。
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer() # 定义一个tf-idf的vectorizer

X = vectorizer.fit_transform([' '.join(seg) for seg in qlist_seg])  # 结果存放在X矩阵

# 测试
def sparsity_ratio(X):
    return 1.0 - X.nnz / float(X.shape[0] * X.shape[1])

print(X.shape)
print("input sparsity ratio:", sparsity_ratio(X))  # 打印出稀疏度(sparsity)



# 对于用户的输入问题，找到相似度最高的TOP5问题，并把5个潜在的答案做返回

from queue import PriorityQueue


def top5results(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """
    q_vector = vectorizer.transform([' '.join(text_preprocessing(input_q))])
    # 计算余弦相似度，tfidf默认使用l2范数；矩阵乘法
    sim = (X * q_vector.T).toarray()

    # 使用优先队列找出top5
    pq = PriorityQueue()
    for cur in range(sim.shape[0]):
        pq.put((sim[cur][0], cur))
        if len(pq.queue) > 5:
            pq.get()

    pq_rank = sorted(pq.queue, reverse=True, key=lambda x: x[0])
    # print([x[0] for x in pq_rank])
    top_idxs = [x[1] for x in pq_rank]  # top_idxs存放相似度最高的（存在qlist里的）问题的下表

    return [alist[i] for i in top_idxs]  # 返回相似度最高的问题对应的答案，作为TOP5答案

# 测试
print(top5results("Which airport was shut down?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results("Which airport is closed?"))
print(top5results("What government blocked aid after Cyclone Nargis?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results("Which government stopped aid after Hurricane Nargis?"))

# 可以明显发现，tfidf方法是很难处理语义相似的问题的。


# 利用倒排表的优化。
#
# 上面的算法，一个最大的缺点是每一个用户问题都需要跟库里的所有的问题都计算相似度。假设我们库里的问题非常多，这将是效率非常低的方法。 这里面一个方案是通过倒排表的方式，先从库里面找到跟当前的输入类似的问题描述。然后针对于这些candidates问题再做余弦相似度的计算。这样会节省大量的时间。

# TODO: 基于倒排表的优化。在这里，我们可以定义一个类似于hash_map, 比如 inverted_index = {}， 然后存放包含每一个关键词的文档出现在了什么位置，
#       也就是，通过关键词的搜索首先来判断包含这些关键词的文档（比如出现至少一个），然后对于candidates问题做相似度比较。
from collections import defaultdict

inverted_idx = defaultdict(set)  # 制定一个一个简单的倒排表
for cur in range(len(qlist_seg)):
    for word in qlist_seg[cur]:
        inverted_idx[word].add(cur)


def top5results_invidx(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    seg = text_preprocessing(input_q)
    candidates = set()
    for word in seg:
        # 取所有包含任意一个词的文档的并集
        candidates = candidates | inverted_idx[word]
    candidates = list(candidates)

    q_vector = vectorizer.transform([' '.join(seg)])
    # 计算余弦相似度，tfidf用的l2范数，所以分母为1；矩阵乘法
    sim = (X[candidates] * q_vector.T).toarray()

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


def top5results_invidx(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    seg = text_preprocessing(input_q)
    candidates = set()
    for word in seg:
        # 取所有包含任意一个词的文档的并集
        candidates = candidates | inverted_idx[word]
    candidates = list(candidates)

    q_vector = vectorizer.transform([' '.join(seg)])
    # 计算余弦相似度，tfidf用的l2范数，所以分母为1；矩阵乘法
    sim = (X[candidates] * q_vector.T).toarray()

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

# 测试
print(top5results_invidx("Which airport was shut down?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results_invidx("Which airport is closed?"))
print(top5results_invidx("What government blocked aid after Cyclone Nargis?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results_invidx("Which government stopped aid after Hurricane Nargis?"))



# 基于词向量的文本表示
#
# 上面所用到的方法论是基于词袋模型（bag-of-words model）。这样的方法论有两个主要的问题：1. 无法计算词语之间的相似度 2. 稀疏度很高。 接下来我们采用词向量作为文本的表示，下载训练好的d=100（100维）的词向量glove.6B.zip。
# TODO
# 读取每一个单词的嵌入。
# 句子向量 = 词向量的平均（出现在问句里的）， 如果给定的词没有出现在词典库里，则忽略掉这个词。
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np

# 将GloVe转为word2vec
_ = glove2word2vec('./data/glove.6B.100d.txt', './data/glove2word2vec.6B.100d.txt')
model = KeyedVectors.load_word2vec_format('./data/glove2word2vec.6B.100d.txt')


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


X = np.zeros((len(qlist_seg), 100))
for cur in range(X.shape[0]):
    X[cur] = docvec_get(qlist_seg[cur])

# 计算X每一行的l2范数
Xnorm2 = np.linalg.norm(X, axis=1, keepdims=True)
X = X / Xnorm2


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


# 测试
print(top5results_emb("Which airport was shut down?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results_emb("Which airport is closed?"))
print(top5results_emb("What government blocked aid after Cyclone Nargis?"))    # 在问题库中存在，经过对比，返回的首结果正确
print(top5results_emb("Which government stopped aid after Hurricane Nargis?"))

# 模型及数据保存
data_dict = dict(inverted_idx=inverted_idx,
                 qlist_seg=qlist_seg,
                 alist=alist,
                 X=X,)

import os
if not os.path.exists('data/model.pk'):
    import pickle
    f = open('data/model.pk', 'wb')
    pickle.dump(data_dict, f)
    f.close()


