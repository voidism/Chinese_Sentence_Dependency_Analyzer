Chinese Sentence Dependency Analyzer
===
Welcome to Dependency Analysis Machine!


## Requirements

- Python3
- gensim==3.4.0
- matplotlib
- numpy
- Download the word2vec models to this folder from [here](https://goo.gl/MeChvA).

## Execute the Program

```python 
python3 Analyser.py
or
python3
>>> from Analyser import *
>>> d = Model()
>>> d.demo()
```

## Four input operation modes:
* Input one word, find the top 20 most similar words & collocation words.
```bash
# first column-> similar words
# second column-> most frequent context words(P(word|word_sent) is high)
# third column -> most frequent center words(P(word_sent|word) is high)

# example:
sent> 漂亮
相似詞前 20 排序              搭配詞前 20 排序              誰的搭配詞是我？前 20 排序
好看,0.7448046207427979       身材,0.5180677771568298       更,0.8341610431671143
耀眼,0.7220890522003174       漂亮,0.5077099800109863       最,0.7704468816518784
搶眼,0.7152576446533203       外型,0.47800153493881226      非常,0.6466405093669891
可愛,0.711075484752655        模樣,0.47470301389694214      很,0.6412381678819656
亮麗,0.7092759013175964       外表,0.4365552216768265       相當,0.5317070707678795
出色,0.6972562670707703       喔,0.4283532500267029         漂亮,0.5077106133103371
好,0.69332355260849           打出,0.4058349132537842       表現,0.5034487694501877
亮眼,0.6865640878677368       看起來,0.40087252855300903    打出,0.48756420612335205
難看,0.677635669708252        成績單,0.38954824209213257    更為,0.46783987432718277
美麗,0.6743546724319458       來得,0.382993221282959        這麼,0.455462783575058
爛,0.6658402681350708         露出,0.3631395101547241       比較,0.45098163187503815
棒,0.6613122224807739         可愛,0.35718709230422974      來得,0.4355574771761894
酷,0.6541733741760254         演技,0.35642683506011963      身材,0.43112702667713165
炫,0.6535170078277588         得多,0.3560766577720642       外表,0.42566750198602676
甜美,0.650082528591156        啦,0.34972161054611206        得多,0.4239025339484215
帥氣,0.6463396549224854       穿上,0.3422245383262634       又,0.3993404284119606
完美,0.6422654390335083       看似,0.33999890089035034      最為,0.3976890444755554
迷人,0.6410719752311707       鏡頭,0.3348468244075775       無比,0.39761487394571304
醜,0.6407166123390198         臉,0.3335908055305481         展現,0.3853464126586914
討喜,0.6300048232078552       美女,0.33281199634075165      頗為,0.38449157029390335
----------------------------
```

* Input two words, compute the cosine similarity of them.

```bash
# first column-> cosine similarity of word_1 and word_2
# second column-> P(word_2|word_1) after scale
# third column -> P(word_1|word_2) after scale

# example:
sent> 緊急 救護
計算 Cosine 相似度            B是A的搭配詞信心值            A是B的搭配詞信心值
0.5115128391230042            0.9036871790885925            0.625317320227623
----------------------------
```

* Input three words, make analogical reasoning.

```bash
# example
sent> 車 飛機 輛
車 to 飛機 == 輛 to ?
架,0.8138582706451416
架次,0.7429684400558472
艘,0.7332526445388794
枚,0.7054634094238281
戰鬥機,0.6152468323707581
運輸機,0.6132361888885498
磅,0.604849100112915
噸,0.5938074588775635
隻,0.5916671752929688
座,0.5915172696113586
英尺,0.5892080664634705
呎,0.58461594581604
戰機,0.5813382863998413
英呎,0.5795311331748962
西西,0.577690839767456
客機,0.5710819959640503
顆,0.5684574842453003
公噸,0.5684502720832825
軍用,0.5664805173873901
噴射機,0.562289297580719
----------------------------
```
* Input four words, analysis the dependency matching.
```bash
sent> 中央 研究院 資訊 科學 研究所
           中央       研究院     資訊       科學       研究所
中央       0.00096    [0.01247]  -0.00541   -0.00879   0.00057
研究院     [0.02117]  0.00394    0.00163    0.0152     0.00665
資訊       -0.00604   -0.00017   0.01791    [0.00671]  0.00211
科學       -0.00168   0.01273    0.00705    0.01941    [0.01775]
研究所     0.00681    0.00708    0.00517    [0.01725]  0.01266
----------------------------
```

## output figure:  
![output_figure](https://i.imgur.com/Y8xdsmX.jpg)

## output latex: 
Also, it will output tex file to _dependency.tex_, you can use it to output pdf file:  

![latex](https://i.imgur.com/CpUQUQN.png)

_Input "exit" to leave._  
_Input "sg" or "cbow" to switch between the CBOW and Skip-gram model._  

## Training data (about 10G)
- 中央研究院平衡語料庫
- Yahoo News
- CNA News (main)

## Online Training with your corpus
- call `Model.online_training(filename="corpus_filename")`
- corpus format: document with words segmented by space.

*************************************************************************
Author: Yung-Sung Chuang 2018/07/27 @ IIS  Academia Sinica           
