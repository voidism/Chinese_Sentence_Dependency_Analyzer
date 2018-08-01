# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging
import json
import numpy as np

import itertools
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import os
import sys
from jexus import Clock
import pickle
# import networkx
# from matplotlib import pylab

def add_frame(art):
    lines=[]
    if art=='':
        return ''
    elif '\r\n' in art:
        lines=art.split('\r\n')
    elif '\n\r' in art:
        lines=art.split('\n\r')
    elif '\r' in art:
        lines=art.split('\r')
    elif '\n' in art:
        lines=art.split('\n')
    else:
        lines.append(art)
        
    #print lines
    longest=0
    for i in range(len(lines)):
        lines[i]=lines[i]
        #print '\n<'+str(i)+'>\n'
        #print lines[i].encode('raw_unicode_escape').count('\u')
        if len(lines[i])+ch_num(lines[i])>longest:
            longest=len(lines[i])+ch_num(lines[i])
            
    for i in range(len(lines)):
        if lines[i]=='':
            lines[i]="**"+"*"*longest
        elif ch_num(lines[i]):
            lines[i]='*'+lines[i].ljust(longest-ch_num(lines[i]))+'*'
        else:
            lines[i]='*'+lines[i].ljust(longest)+'*'

    lines.insert(0,'*'*longest+"**")
    lines.append('*' * longest + '**')
    return '\n'.join(lines)


def plot_confusion_matrix(cm, classes, target_mat,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues", sv=False, sg=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    font_path = './bkai00mp.ttf'
    prop = mfm.FontProperties(fname=font_path)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,
               fontproperties=prop)
    plt.yticks(tick_marks, classes,
               fontproperties=prop)

    fmt = '.4f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cor = "white" if cm[i, j] > thresh else "black"
        if target_mat[i][j] == 1:
            cor = 'red'
        elif i == j:
            cor = 'green'
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=7,
                 color=cor)

    plt.tight_layout()
    plt.ylabel('Center word')
    plt.xlabel('Context word')
    if sv:
        emb = 'skip_gram' if sg else 'CBOW'
        if not os.path.exists('./dependency_graphs'):
            os.makedirs('./dependency_graphs')
        plt.savefig('./dependency_graphs/{}_{}.png'.format(''.join(classes), emb))
    else:
        plt.show()
    plt.clf()



def ch_num(string):
    ch_n = 0
    for i in string:
        if (i >= u'\u4e00' and i <= u'\u9fa5') or i == '，' or i == '。':
            ch_n += 1
    return ch_n

def comp(string, num):
    ch_n = 0
    for i in string:
        if i >= u'\u4e00' and i <= u'\u9fa5':
            ch_n += 1
    rem = num-len(string)-ch_n
    if rem > 0:
        return string + ' ' * rem
    else:
        return string


def load_cna():
    # li = json.load(open('cna_cmn_SegSyable_cutted_1.json'))['data'][:500]
    # new = [x[0] for x in li]
    # with open('cna_test.json', 'w') as fp:
    #     json.dump({'data':new}, fp)
    return json.load(open('cna_test.json'))['data']



class Model():
    def __init__(self):
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.model = [models.Word2Vec.load('CBOW/word2vec.model'),models.Word2Vec.load('skip-gram/word2vec.model')]
        # self.word2idx = [json.load(open('CBOW/word2idx.json')), json.load(open('CBOW/word2idx.json'))]
        self.sg = 0
        self.art = ''
        self.load_art()
        self.save_fig = False
        self.texfile = open('dependency.tex', 'w', encoding='utf8')
        self.begin_tex()
        
    def __del__(self):
        self.texfile.write(r"\end{CJK*}")
        self.texfile.write('\n')
        self.texfile.write(r"\end{document}")
        self.texfile.close()

        # self.show_pic = True

    def word2idx(self, word):
        return self.model[self.sg].wv.vocab[word].index

    def find_my_collocation(self, word, lim=10000, topn=20):
        if self.sg == 1:
            b = self.model[self.sg][word]
            a = self.model[self.sg].syn1neg[:lim,:]
            sim_array = np.matmul(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))
            order = np.argsort(-sim_array)[:topn]
            ans = []
            for i in order:
                ans.append((self.model[self.sg].wv.index2word[i], sim_array[i]))
            return ans
        else:
            return self.model[self.sg].similar_by_vector(self.model[self.sg].syn1neg[self.word2idx(word)], topn=topn, restrict_vocab=lim)

    def find_whose_collocation_is_me(self, word, lim=10000, topn=20):
        if self.sg == 1:
            return self.model[self.sg].similar_by_vector(self.model[self.sg].syn1neg[self.word2idx(word)], topn=topn, restrict_vocab=lim)
        else:
            b = self.model[self.sg][word]
            a = self.model[self.sg].syn1neg[:lim,:]
            sim_array = np.matmul(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))
            order = np.argsort(-sim_array)[:topn]
            ans = []
            for i in order:
                ans.append((self.model[self.sg].wv.index2word[i], sim_array[i]))
            return ans

    def test_collocation(self, w1, w2):  #examine if w2 is w1's collocation word
        v1 = None
        v2 = None
        res = -1
        try:
            if self.sg:
                v2 = self.model[self.sg].syn1neg[self.word2idx(w2)]
                v1 = self.model[self.sg][w1]
            else:
                v1 = self.model[self.sg].syn1neg[self.word2idx(w1)]
                v2 = self.model[self.sg][w2]
            res = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except:
            pass

        return res
    
    def context_prob(self, w1, w2):
        res = 0
        try:
            v1 = self.model[self.sg][w1]
            v2 = self.model[self.sg].syn1neg[self.word2idx(w2)]
            res = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except:
            res = -1
        return res

    def sent_sim(self, sent):  # s='家 停 區 施'):
        # sent = s.split(' ')
        ln = len(sent)
        mat = np.zeros((ln, ln))
        for i in range(len(sent)):
            for j in range(len(sent)):
                mat[i, j] = self.test_collocation(sent[i], sent[j])
        return mat

    def tag_mat(self, mat):
        target = np.zeros_like(mat)
        if mat.shape[0] == 1:
            return target
        for i in range(mat.shape[0]):
            rank = np.argsort(-mat[i])
            top = rank[0] if rank[0] != i else rank[1]
            for x in range(mat.shape[1]):
                boo = int(x == top)
                if boo:
                    target[i][x] = 1
        return target


    def run_DA(self,big=0,lim=50):
        li = load_cna()[big:lim]
        ct = Clock(len(li),5,'DA')
        log = []
        for i in li:
            temp_dict1 = {'sentence':''.join(i)}
            self.sg = 0
            cm = self.sent_sim(i)
            temp_dict1['graph'] = cm.tolist()
            temp_dict1['sg'] = self.sg
            tag = self.tag_mat(cm)
            log.append(temp_dict1)
            self.texfile.write('\\textbf{CBOW} \\\\ \n')
            self.write_tex(i,tag,cm)
            # plot_confusion_matrix(cm, i, tag, sv=True, sg=self.sg)
            temp_dict2 = {'sentence':''.join(i)}
            self.sg = 1
            cm = self.sent_sim(i)
            temp_dict2['graph'] = cm.tolist()
            temp_dict2['sg'] = self.sg
            tag = self.tag_mat(cm)
            log.append(temp_dict2)
            self.texfile.write('\\textbf{Skip-gram} \\\\ \n')
            self.write_tex(i,tag,cm)
            # plot_confusion_matrix(cm, i, tag, sv=True, sg=self.sg)
            ct.flush()
        with open('log.json', 'w') as fp:
            json.dump({'data':log}, fp)

    def dependency_analysis(self, q_list, sv=1):
        mat = self.sent_sim(q_list)
        target = np.zeros((len(q_list), len(q_list)))
        w = 11
        print(' '*w, end='')
        for i in range(len(q_list)):
            print(comp(q_list[i], w), end='')
        print('')
        for i in range(len(q_list)):
            print(comp(q_list[i], w), end='')
            rank = np.argsort(-mat[i])
            #np.concatenate((np.zeros((i,)), np.argsort(-mat[i][i:])))#
            top = rank[0] if rank[0] != i else rank[1]
            for x in range(len(mat[i])):
                boo = int(x == top)
                txt = '[' * boo + \
                    str(round(mat[i][x], 5)) + ']' * boo
                if boo:
                    target[i][x] = 1
                print(comp(txt, w), end='')
            print('')
        plot_confusion_matrix(
            mat, q_list, target, title='Dependency Analysis', sg=self.sg, sv=sv)
        self.write_tex(q_list,target,mat)


    def demo(self):
        print(add_frame(self.art))
        while True:
            print('sent> ', end='')
            try:
                query = input()
                q_list = query.split()

                if len(q_list) == 1:
                    if q_list[0] == '':
                        continue
                    if q_list[0] == 'exit':
                        del self.model
                        break
                    if q_list[0] == 'sg':
                        self.sg = 1
                        continue
                    if q_list[0] == 'cbow':
                        self.sg = 0
                        continue
                    col = self.find_my_collocation(q_list[0])
                    who = self.find_whose_collocation_is_me(q_list[0])
                    print(comp("相似詞前 20 排序", 30)+comp("搭配詞前 20 排序", 30)+comp("誰的搭配詞是我？前 20 排序", 30))
                    res = self.model[self.sg].most_similar(q_list[0], topn=20)
                    for item, con, man in zip(res, col, who):
                        print(comp(item[0]+","+str(item[1]), 30) +
                              comp(con[0] + "," + str(20*con[1] if self.sg else 40*con[1]), 30) +
                              comp(man[0] + "," + str(20*man[1] if self.sg else 40*man[1]), 30))

                elif len(q_list) == 2:
                    print(comp("計算 Cosine 相似度", 30)+comp("B是A的搭配詞信心值", 30)+comp("A是B的搭配詞信心值", 30))
                    res = self.model[self.sg].similarity(q_list[0], q_list[1])
                    res1 = self.test_collocation(q_list[0], q_list[1])
                    res2 = self.test_collocation(q_list[1], q_list[0])
                    print(comp(str(res), 30) +
                          comp(str(20*res1 if self.sg else 40*res1), 30) +
                          comp(str(20*res2 if self.sg else 40*res2), 30))

                elif len(q_list) == 3:
                    print("%s to %s == %s to ?" %
                          (q_list[0], q_list[1], q_list[2]))
                    # w1 = self.model[self.sg][q_list[0]]
                    # w2 = self.model[self.sg][q_list[1]]#positive=[w2, w3], negative=[w1]
                    # w3 = self.model[self.sg][q_list[2]]
                    res = self.model[self.sg].most_similar(positive=[q_list[1], q_list[2]], negative=[q_list[0]], topn=20)
                    for item in res:
                        print(item[0] + "," + str(item[1]))
                else:
                    self.dependency_analysis(q_list, sv=0)
                print("----------------------------")
            except Exception as e:
                print(repr(e))
                print("----------------------------")

    def load_art(self):
        self.art = ''' ============ Welcome to Dependency Analysis Machine ============ 

Four input operation mode: \nInput one word, find the top 20 most similar words & collocation words.
Input two words, compute the cosine similarity of them.
Input three words, make analogical reasoning.
Input four words, analysis the dependency matching.

Input "exit" to leave.
Input "sg" or "cbow" to switch between the CBOW and Skip-gram model.

Author: Yung - Sung Chuang 2018 / 07 / 27 @ IIS Academia Sinica '''

    def begin_tex(self):
        begin = r'''\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{CJKutf8}
\usepackage{tikz-dependency}
\usepackage[left=0.5in, right=0.5in, bottom=0.5in, top=0.5in]{geometry}
\begin{document}
\begin{CJK*}{UTF8}{bsmi}'''
        self.texfile.write(begin)
        self.texfile.write('\n')

    def write_tex(self, sent, tag_mat, score_mat):
        self.texfile.write(r"""\begin{dependency}
    \begin{deptext}[column sep=1.2e m] """)
        self.texfile.write('\n        ')
        self.texfile.write(r' \& '.join(sent) + r' \\')
        self.texfile.write('\n    \\end{deptext}\n')
        # self.texfile.write('    \\deproot{4}{root}\n')
        for i in range(tag_mat.shape[0]):
            for j in range(tag_mat.shape[1]):
                if tag_mat[i,j] == 1:
                    self.texfile.write("        \\depedge{%d}{%d}{%s}\n"%(i+1,j+1,str(round(score_mat[i,j],4))))#str(i)+"-"+str(j)
        self.texfile.write("\\end{dependency}\n\n")



if __name__ == "__main__":
    model = Model()
    model.demo()
