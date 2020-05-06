# -*- encoding:utf-8 -*-
import jieba
import jieba.posseg as pseg
import codecs
import csv
stopwords=['吕州','林城','银行卡','明白','白云','嗡嗡嘤嘤',
           '阴云密布','雷声','陈大','谢谢您','安置费','任重道远',
           '孤鹰岭','阿庆嫂','岳飞','师生','养老院','段子','老总']
replace_words={'师母':'吴慧芬','陈老':'陈岩石','老赵':'赵德汉','达康':'李达康','高总':'高小琴',
              '猴子':'侯亮平','老郑':'郑西坡','小艾':'钟小艾','老师':'高育良','同伟':'祁同伟',
              '赵公子':'赵瑞龙','郑乾':'郑胜利','孙书记':'孙连城','赵总':'赵瑞龙','昌明':'季昌明',
               '沙书记':'沙瑞金','郑董':'郑胜利','宝宝':'张宝宝','小高':'高小凤','老高':'高育良',
               '伯仲':'杜伯仲','老杜':'杜伯仲','老肖':'肖钢玉','刘总':'刘新建',"美女老总":"高小琴"}
names={} #姓名字典
relationships ={} #关系字典
lineNames =[] #每段内人物的关系
node=[] #存放处理后的人物
def read_txt(path): #读取剧作并分词
    jieba.load_userdict("人民的名义.txt") #加载人物字典(注意这个文件要用utf-8编码，可以使用sublime进行转换为utf-8编码)
    f=codecs.open(path,'rb') #读取剧作,并将其转换为utf-8编码
    for line in f.readlines():
        poss=pseg.cut(line)  #分词并返回该词词形
        lineNames.append([])  #为新读入的一段添加人物名称列表
        for w in poss:
            if w.word in stopwords:  #去掉某些停用词
                continue
            if w.flag != "nr" or len(w.word) <2 :
                if w.word not in replace_words:
                    continue
            if w.word in replace_words: #将某些在文中人物的昵称替换成正式的名字
                w.word=replace_words[w.word]
            lineNames[-1].append(w.word)  #为当前段增加一个人物
            if names.get(w.word) is None: #如果这个名字从来没出现过，初始化这个名字
                names[w.word] =0
                relationships[w.word] ={}
            names[w.word] +=1 #该人物出现次数加1
    for line in lineNames: #通过对于每一段段内关系的累加，得到在整篇小说中的关系
        for name1 in line:
            for name2 in line:
                if name1 == name2:
                    continue
                if relationships[name1].get(name2) is None: #如果没有出现过两者之间的关系，则新建项
                    relationships[name1][name2] =1
                else:
                    relationships[name1][name2] +=1 #如果两个人已经出现过，则亲密度加1
def write_csv():
    csv_edge_file = open("edge.csv", "w", newline="")
    writer = csv.writer(csv_edge_file)
    writer.writerow(["source", "target", "weight","type"])  # 先写入列名,"type"为生成无向图做准备
    for name,edges in relationships.items():
        for v,w in edges.items():
            if w>20:
                node.append(name)
                writer.writerow((name,v,str(w),"undirected"))  # 按行写入数据
    csv_edge_file.close()
    #生成node文件
    s=set(node)
    csv_node_file =open("node.csv","w",newline="")
    wnode =csv.writer(csv_node_file)
    wnode.writerow(["ID","Label","Weight"])
    for name,times in names.items():
        if name in s:
            wnode.writerow((name,name,str(times) ) )
    csv_node_file.close()

if __name__=='__main__':
    file = "人民的名义.txt"
    edge_file="edge.txt"
    read_txt(file)
    write_csv()