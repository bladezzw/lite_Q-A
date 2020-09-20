print('Loading model, Please wait.')
from use_Glove import top5results_emb

if __name__ == '__main__':

    ques = ''
    while ques != 'q':
        print('Please input your question:(e.g.  what is classic music?)')
        ques = input()
        print(top5results_emb(ques))
