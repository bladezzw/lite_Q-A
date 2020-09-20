# 理解数据（可视化分析/统计信息）
#
# 对数据的理解是任何AI工作的第一步，需要充分对手上的数据有个更直观的理解。
# TODO: 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
#       把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
#       需要使用matplotlib里的plot函数。y轴是词频
from data_process import read_corpus

from collections import Counter
import matplotlib.pyplot as plt

qlist, alist = read_corpus('./data/train-v2.0.json')

words_cnt = Counter()
for text in qlist:  # 统计词频
    words_cnt.update(text.strip(' .!?').split())

value_sort = sorted(words_cnt.values(), reverse=True)
plt.subplot(221)
plt.plot(value_sort)
plt.subplot(222)
plt.plot(value_sort[:2000])
plt.subplot(223)
plt.plot(value_sort[:200])
plt.subplot(224)
plt.plot(value_sort[:20])
plt.show()

# 显示词频最高前10词，因为只取高频值，所以value转换时重合的概率较小，即时重合也没有太大影响
inverse = dict(zip(words_cnt.values(), words_cnt.keys()))
print("词数(type)：%d" % len(words_cnt))
print([[inverse[v], v] for v in value_sort[:20]])