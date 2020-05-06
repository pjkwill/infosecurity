#coding=utf-8
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
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection  import cross_val_score

# 获得所有文件的相对路径和标签
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

 # 对文档中的每一行进行中文分词，结果为返回一行字符串代表单个样本的特征集合
def segDepart(sentence):
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

# 将目录下文件特征提取出来，并存为一个文件，一行代表一个文件提取结果，列为分词结果
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

 #寻找最佳主成分数量，还原度取适宜值，这里取95%处x值，过大数据噪点影响明显
def findPCA(weight):
    candidate_components = range(10, 300, 30)
    explained_ratios = []
    for c in candidate_components:
        pca = PCA(n_components=c)
        X_pca = pca.fit_transform(weight)
        explained_ratios.append(np.sum(pca.explained_variance_ratio_)) #计算对原材料的还原度
    plt.figure(figsize=(10, 6), dpi=144)
    plt.grid()
    plt.plot(candidate_components, explained_ratios)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained variance ratio for PCA')
    plt.yticks(np.arange(0.5, 1.05, .05))
    plt.xticks(np.arange(0, 300, 20))
    plt.show()

# 寻找KNN最佳参数，误差值y越小效果越好
def find_n_neighbors(x, y):
    k_range = range(1, 31)
    k_error = []
    #循环，取k=1到k=31，查看误差效果
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')
        k_error.append(1 - scores.mean())

    #画图，x轴为k值，y值为误差值
    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()

# 训练KNN模型并进行测试
def getModel(labels):
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
    #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

    # findPCA(weight) ## 依据图像找到最佳PCA参数:160
    pca = PCA(n_components=160)
    all_pca = pca.fit_transform(weight)
    # find_n_neighbors(all_pca, labels) ##依据图像找到KNN最佳取值:12
    train_pca = all_pca[:200]
    test_pca = all_pca[200:]
    train_label = labels[:200]
    test_label = labels[200:]
     #定义一个knn分类器对象
    knn = KNeighborsClassifier(n_neighbors=12) 
   #调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
    knn.fit(train_pca, train_label)  
    
    y_predict = knn.predict(test_pca) 
    score=knn.score(test_pca,test_label,sample_weight=None)
    #输出原有标签
    print('y_predict = ')  
    print(y_predict)  
    #输出测试的结果
    print('y_test = ')
    print(test_label)    
    #输出准确率
    print ('Accuracy:',score  )




def main():
    # 将数据保存到文件中，可快速验证
    # files, labels = getFiles()
    # np.save('files.npy', files)
    # np.save('labels.npy', labels)

    #files = np.load('files.npy')
    labels = np.load('labels.npy')
    #saveInOne(files)
    getModel(labels)



if __name__ == '__main__':
    main()
