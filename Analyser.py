# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging
import json
import numpy as np

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import os
import sys
from jexus import Clock
import pickle
import networkx
from matplotlib import pylab

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
    
    def word2idx(self, word):
        return self.model[self.sg].wv.vocab[word].index

    def find_collocation(self, word, lim=10000):
        return self.model[self.sg].similar_by_vector(self.model[self.sg].syn1neg[self.word2idx(word)], topn=20, restrict_vocab=lim)

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
                mat[i, j] = self.context_prob(sent[i], sent[j])
        return mat

    def tag_mat(self, mat):
        target = np.zeros_like(mat)
        if mat.shape[0] == 1:
            return target
        for i in range(len(mat)):
            rank = np.argsort(-mat[i])
            top = rank[0] if rank[0] != i else rank[1]
            for x in range(len(mat[i])):
                boo = int(x == top)
                if boo:
                    target[i][x] = 1
        return target


    def run_DA(self):
        li = load_cna()
        ct = Clock(len(li),5,'DA')
        log = []
        for i in li:
            temp_dict1 = {'sentence':''.join(i)}
            self.sg = 0
            cm = self.sent_sim(i)
            temp_dict1['graph'] = cm.tolist()
            temp_dict1['sg'] = self.sg
            # tag = self.tag_mat(cm)
            log.append(temp_dict1)
            # plot_confusion_matrix(cm, i, tag, sv=True, sg=self.sg)
            temp_dict2 = {'sentence':''.join(i)}
            self.sg = 1
            cm = self.sent_sim(i)
            temp_dict2['graph'] = cm.tolist()
            temp_dict2['sg'] = self.sg
            # tag = self.tag_mat(cm)
            log.append(temp_dict2)
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

    def draw_dependency(self):
        g = dg.nx_graph()
        g.info()
        pos = networkx.spring_layout(g, dim=1)
        networkx.draw_networkx_nodes(g, pos, node_size=50)
        # networkx.draw_networkx_edges(g, pos, edge_color='k', width=8)
        networkx.draw_networkx_labels(g, pos, dg.nx_labels)
        pylab.xticks([])
        pylab.yticks([])
        pylab.savefig('tree.png')
        pylab.show()

    def demo(self):
        # print("* ======== Welcome to Depandency Analysis Machine ======== *")
        # print("提供 4 種測試模式\n")
        # print("輸入一個詞，則去尋找前20個該詞的相似詞 & 鄰近詞")
        # # print("輸入一個詞+，則去尋找前20個該詞的鄰近詞")
        # print("輸入兩個詞，則去計算兩個詞的餘弦相似度")
        # print("輸入三個詞，進行類比推理")
        # print("輸入四個以上的詞，進行文字匹配\n")
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
                        break
                    if q_list[0] == 'sg':
                        self.sg = 1
                        continue
                    if q_list[0] == 'cbow':
                        self.sg = 0
                        continue
                    col = self.find_collocation(q_list[0])
                    print(comp("相似詞前 20 排序", 30)+"鄰近詞前 20 排序")
                    res = self.model[self.sg].most_similar(q_list[0], topn=20)
                    for item, con in zip(res, col):
                        print(comp(item[0]+","+str(item[1]), 30) +
                              '\t'+con[0] + "," + str(con[1]))

                elif len(q_list) == 2:
                    print("計算 Cosine 相似度")
                    res = self.model[self.sg].similarity(q_list[0], q_list[1])
                    print(res)
                elif len(q_list) == 3:
                    print("%s to %s == %s to ?" %
                          (q_list[0], q_list[1], q_list[2]))
                    w1 = self.model[self.sg][q_list[0]]
                    w2 = self.model[self.sg][q_list[1]]
                    w3 = self.model[self.sg][q_list[2]]
                    res = self.model[self.sg].similar_by_vector((w2 - w1) + w3, topn=20)
                    # print("%s之於%s，如%s之於" % (q_list[0],q_list[2],q_list[1]))
                    # res = model.most_similar([q_list[0],q_list[1]], [q_list[2]], topn= 20)
                    for item in res:
                        print(item[0] + "," + str(item[1]))
                else:
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
                        mat, q_list, target, title='Dependency Analysis', sg=self.sg, sv=self.save_fig)
                print("----------------------------")
            except Exception as e:
                print(repr(e))
                print("----------------------------")

    # def plot_confusion_matrix(self, cm, classes, target_mat,
    #                       normalize=False,
    #                       title='Confusion matrix',
    #                       cmap="Blues"):
    #     """
    #     This function prints and plots the confusion matrix.
    #     Normalization can be applied by setting `normalize=True`.
    #     """
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    #     else:
    #         print('Confusion matrix, without normalization')

    #     # print(cm)

    #     font_path = './bkai00mp.ttf'
    #     prop = mfm.FontProperties(fname=font_path)
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45,
    #             fontproperties=prop)
    #     plt.yticks(tick_marks, classes,
    #             fontproperties=prop)

    #     fmt = '.4f'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         cor = "white" if cm[i, j] > thresh else "black"
    #         if target_mat[i][j] == 1:
    #             cor = 'red'
    #         elif i == j:
    #             cor = 'green'
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                 horizontalalignment="center", fontsize=7,
    #                 color=cor)

    #     plt.tight_layout()
    #     plt.ylabel('Center word')
    #     plt.xlabel('Context word')
    #     if self.save_fig:
    #         emb = 'skip_gram' if self.sg else 'CBOW'
    #         if not os.path.exists('./dependency_graphs'):
    #             os.makedirs('./dependency_graphs')
    #         plt.savefig('./dependency_graphs/{}_{}.png'.format(''.join(classes), emb))
    #     else:
    #         plt.show()

    def load_art(self):
        self.art = ''' ============ Welcome to Dependency Analysis Machine ============ 

Four input operation mode: \nInput one word, find the top 20 most similar words & collocation words.
Input two words, compute the cosine similarity of them.
Input three words, make analogical reasoning.
Input four words, analysis the dependency matching.

Input "exit" to leave.
Input "sg" or "cbow" to switch between the CBOW and Skip-gram model.

Author: Yung-Sung Chuang 2018/07/27 @ IIS  Academia Sinica'''


if __name__ == "__main__":
    model = Model()
    model.demo()
