#coding=utf-8
'''
因为没有很好解决聚类标签和原来标签如何对应的问题
所以只能将实验模型和矩阵存为文件，手动对比
训练集数量：200
测试集数量: 72

'''
import os
import random
import numpy as np
import jieba
import re
import pandas as pd
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.cluster import KMeans


def getFiles():
    baseDir = './/database'
    firstDirList = [os.path.join(baseDir, f)
                    for f in os.listdir(baseDir)]  #获得一级目录列表
    filePath = []
    labelList = []  #原始标签列表，用于测试结果对比
    for file in firstDirList:  #处理二级目录，获得具体文件路径
        label = file.split('-')[0][13:]
        secondDirList = [os.path.join(file, f) for f in os.listdir(file)]
        # print(len(secondDirList))
        if len(secondDirList) > 100:
            secondDirList = secondDirList[:50]  #如果数量过多则取50个
        tmp = [label] * len(secondDirList)
        labelList.extend(tmp)
        filePath.extend(secondDirList)

    randnum = os.urandom(8)  
    random.seed(randnum)
    random.shuffle(labelList) #以同样方式对路径和标签打乱
    random.seed(randnum)
    random.shuffle(filePath)
    return filePath, labelList


def segDepart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    newSentence = re.sub(r'[^\u4e00-\u9fa5]', ' ', sentence)  #只保留中文
    sentenceDepart = jieba.cut(newSentence.strip())
    # 创建一个停用词列表
    stopwords = [
        line.strip()
        for line in open('stopwords.txt', encoding='UTF-8').readlines()
    ]
    outstr = ''
    # 去停用词，拼接词为一行字符串
    for word in sentenceDepart:
        if word not in stopwords:
            if word != '\t':
                if len(word) >= 2:
                    outstr += word
                    outstr += " "
    return outstr


def saveInOne(path):
    outputs = open("out.txt", 'w', encoding='UTF-8')
    for filename in path:
        inputs = open(filename, 'r', encoding='gb18030', errors='ignore')
        for line in inputs:
            lineSeg = segDepart(line)
            outputs.write(lineSeg)
        outputs.write('\n') #一篇文章的分词内容占一行
        inputs.close()
    outputs.close()


def getModel():

    corpus = []

    #读取预料 一行预料为一个文档
    for line in open('out.txt', 'r', encoding='UTF-8').readlines():
        corpus.append(line.strip())

    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(min_df=10)

    #该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    #获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    #打印特征向量文本内容
    resName = "Tfidf_Result.txt"
    result = codecs.open(resName, 'w', 'utf-8')  #不易出现编码问题
    for j in range(len(word)):
        result.write(word[j] + ' ')
    result.write('\n')

    #每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        for j in range(len(word)):
            result.write(str(weight[i][j]) + ' ')
        result.write('\n')
    result.close()

    print('Start Kmeans:')

    clf = KMeans(n_clusters=6) #已知可分为6类
    s = clf.fit(weight)
    joblib.dump(clf, 'km.pkl')

    #每个样本所属的簇
    label = []
    for i in range(len(clf.labels_)):
        label.append(clf.labels_[i])
    print(label)

    y_pred = clf.labels_
    pca = PCA(n_components=2)  #输出两维
    newData = pca.fit_transform(weight)  #载入N维

    xs, ys = newData[:, 0], newData[:, 1]
    #设置颜色
    cluster_colors = {
        0: 'r',
        1: 'yellow',
        2: 'b',
        3: 'chartreuse',
        4: 'purple',
        5: '#FFC0CB',
        6: '#6A5ACD',
        7: '#98FB98'
    }

    #设置类名
    cluster_names = {
        0: u'类0',
        1: u'类1',
        2: u'类2',
        3: u'类3',
        4: u'类4',
        5: u'类5',
        6: u'类6',
        7: u'类7'
    }

    df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus))
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(8, 5))  # set size
    ax.margins(0.02)
    for name, group in groups:
        ax.plot(
            group.x,
            group.y,
            marker='o',
            linestyle='',
            ms=10,
            label=cluster_names[name],
            color=cluster_colors[name],
            mec='none')
    plt.show()
    return label


def testFunc(path, label):
    outputs = open("test_out.txt", 'w', encoding='UTF-8')
    for filename in path:
        inputs = open(filename, 'r', encoding='gb18030', errors='ignore')
        for line in inputs:
            lineSeg = segDepart(line)
            outputs.write(lineSeg) #一篇测试集文章的分词内容占一行，写入文档
        outputs.write('\n')
        inputs.close()

    model_data = codecs.open('Tfidf_Result.txt', 'r', 'utf-8')
    model_output = model_data.readline()  #获取模型的第一行词信息，代表所有维度
    model_data.close()
    outputs.write(model_output)
    outputs.close()

    corpus = []

    #读取预料 一行预料为一个文档
    for line in open("test_out.txt", 'r', encoding='UTF-8').readlines():
        corpus.append(line.strip())

    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(min_df=10)

    #该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    #获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()

    #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    test_weight = tfidf.toarray()

    # 载入保存的模型
    clf = joblib.load('km.pkl')

    clf.fit_predict(test_weight)
    testRes = []
    for i in range(len(clf.labels_)):
        testRes.append(clf.labels_[i])
    return testRes


def main():
    # files, labels = getFiles()
    #np.save('files.npy', files)
    #np.save('labels.npy', labels)
    files = np.load('files.npy')
    labels = np.load('labels.npy')
    trainData = files[:200]
    trainLabel = labels[:200]
    testData = files[200:]
    testLabel = labels[200:]
    saveInOne(trainData)
    theLabel = getModel()
    np.save('resTrainLabel.npy', theLabel)  #模型预测的训练集标签
    theLabel = testFunc(testData, testLabel)
    np.save('resTestLabel.npy', theLabel)  #模型预测的测试集标签
    np.save('testLabel.npy', testLabel)  #实际测试集标签

#暂时未找到聚类标签与原有标签的自动对应关系，需要人工，L1:模型聚类结果 L2：真实标签
def computeError(L1, L2):  
    mydict = {'7': 1, '39': 5, '4': 3, '34': 0, '5': 2, '17': 4}
    #计算准确率
    rightCnt = 0

    for i in range(len(L2)): #列表元素一一对应
        tmp = mydict.get(L2[i])
        if tmp == L1[i]:
            rightCnt = rightCnt + 1
    accuracy = rightCnt / len(L2)

    return accuracy


if __name__ == '__main__':
    #main()
    labels = np.load('labels.npy')
    trainLabel = labels[:200]
    resTrainLabel = np.load('resTrainLabel.npy')
    resTestLabel = np.load('resTestLabel.npy')
    # print(labels)
    # print(resTrainLabel)
    # print(resTestLabel)
    # for i in range(len(resTrainLabel)):
    #     print(trainLabel[i],"-->",resTrainLabel[i])
    accu = computeError(resTestLabel, labels[200:])
    print("%f" % accu)
