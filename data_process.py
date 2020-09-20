import json

"""
读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分要在后面处理）
qlist = ["问题1"， “问题2”， “问题3” ....]
alist = ["答案1", "答案2", "答案3" ....]
务必要让每一个问题和答案对应起来（下标位置一致）
"""
def read_corpus(filepath):
    with open(filepath) as f:
        data = json.load(f)

    qlist = list()
    alist = list()
    for item in data['data']:
        for para in item['paragraphs']:
            for qa in para['qas']:
                qlist.append(qa['question'])
                # 部分answers的list为空，所以会引发IndexError
                try:
                    alist.append(qa['answers'][0]['text'])
                except IndexError:
                    qlist.pop()

    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist
if __name__ == '__main__':
    # 测试
    qlist, alist = read_corpus('./data/train-v2.0.json')
    print("问答数量：%d" % len(qlist))
    print(qlist[-3:])
    print(alist[-3:])