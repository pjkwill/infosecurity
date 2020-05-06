#！user/bin/env python
#-*- coding utf-8 -*-
# author:LiRuikun
#！user/bin/env python
#-*- coding utf-8 -*-
# author:LiRuikun


import codecs
import jieba.posseg as pseg
import jieba

names = {}#  保存人物，键为人物名称，值为该人物在全文中出现的次数
relationships = {}#保存人物关系的有向边，键为有向边的起点，值为一个字典 edge ，edge 的键为有向边的终点，值是有向边的权值
lineNames = []# 缓存变量，保存对每一段分词得到当前段中出现的人物名称

jieba.load_userdict("names.txt")#加载人物表
with codecs.open("红楼梦.txt", 'r', 'utf8') as f:
    for line in f.readlines():
        poss = pseg.cut(line)  # 分词，返回词性
        lineNames.append([])  # 为本段增加一个人物列表
        for w in poss:
            if w.flag != 'nr' or len(w.word) < 2:
                continue  # 当分词长度小于2或该词词性不为nr（人名）时认为该词不为人名
            lineNames[-1].append(w.word)  # 为当前段的环境增加一个人物
            if names.get(w.word) is None:  # 如果某人物（w.word）不在人物字典中
                names[w.word] = 0
                relationships[w.word] = {}
            names[w.word] += 1

for line in lineNames:
    for name1 in line:
        for name2 in line:
            if name1 == name2:
                continue
            if relationships[name1].get(name2) is None:
                relationships[name1][name2] = 1
            else:
                relationships[name1][name2] = relationships[name1][name2] + 1


#边
with codecs.open("People_node.txt", "w", "utf8") as f:
    f.write("ID Label Weight\r\n")
    for name, times in names.items():
        if times > 10:
            f.write(name + " " + name + " " + str(times) + "\r\n")


#节点
with codecs.open("People_edge.txt", "w", "utf8") as f:
    f.write("Source Target Weight\r\n")
    for name, edges in relationships.items():
        for v, w in edges.items():
            if w > 10:
                f.write(name + " " + v + " " + str(w) + "\r\n")



f=open('People_edge.txt','r',encoding='utf-8')
f2=open('names.txt','r',encoding='utf-8').read()
lines=f.readlines()

