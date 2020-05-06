#coding=utf-8
import os
from collections import Counter
import numpy as np
import random
import re
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']


def getFileDiv():
    hamFile = './/email//ham'
    spamFile = './/email//spam'
    hamEmails = [os.path.join(hamFile,f) for f in os.listdir(hamFile)]    
    spamEmails = [os.path.join(spamFile,f) for f in os.listdir(spamFile)]
    label = np.zeros(50)
    label[len(hamEmails):50] = 1
    hamEmails.extend(spamEmails)
    randnum = os.urandom(8) #以同样方式对路径和标签打乱
    random.seed(randnum)
    random.shuffle(label)
    random.seed(randnum)
    random.shuffle(hamEmails)
    return hamEmails,label


def getWordsProb(filepath, label):
    spamList = []
    hamList = []
    for index,path in enumerate(filepath):
        with open(path, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub('[^a-zA-Z]',' ',line)
                words = line.split()
                if label[index] == 0:
                    hamList.extend(words)
                else:
                    spamList.extend(words)
    spamCounter = Counter(spamList)
    hamCounter = Counter(hamList)
    for item in list(spamCounter): #处理单字符
        if len(item) == 1:
            del spamCounter[item]
    for item in list(hamCounter):
        if len(item) == 1:
            del hamCounter[item]
    spamSet = set(spamCounter)
    hamSet = set(hamCounter)
    spamSet.update(hamSet)
    allWordList = Counter(spamSet)

    spamDict = {}
    hamDict = {}
    spamCounter = allWordList + spamCounter #消除概率为零相乘的影响
    spamCnt = sum(spamCounter.values()) 
    for k,v in spamCounter.items():
        spamDict[k] = v/spamCnt
    hamCounter = allWordList + hamCounter
    hamCnt = sum(hamCounter.values())
    for k,v in hamCounter.items():
        hamDict[k] = v/hamCnt
    #print(sum(hamDict.values()), sum(spamDict.values()))
    return hamDict,spamDict

def mulNBTest(hamDict, spamDict, testEmail, testLabel):
    result = [] #记录判断结果，之后与Label对比
    spamProb = 0.5  #P(spam) = 0.5
    hamProb = 0.5  #P(ham) = 0.5
    for testFile in testEmail:
        testWords = []
        with open(testFile, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = re.sub('[^a-zA-Z]',' ',line)
                words = line.split()
                testWords.extend(words)

        testCounter = Counter(testWords)
        for item in list(testCounter): #处理单字符
            if len(item) == 1:
                del  testCounter[item]
        pureWords = list(testCounter) #得到邮件内字符列表
        probList = []  #存储每个字符的贡献
        mediumFre1 = np.median(list(hamDict.values()))
        mediumFre2 = np.median(list(spamDict.values()))
        for word in pureWords:
            pwh = hamDict.get(word, mediumFre1)  # P(word|ham)
            pws = spamDict.get(word, mediumFre2)  # P(word|spam)
            psw = (spamProb*pws)/(pwh*hamProb+pws*spamProb) # P(spam|word) = P(spam)*P(word|spam)/P(word)
            probList.append(psw)
        numerator = 1  #分子
        denominator= 1  #分母
        for psw in probList:
            numerator *= psw
            denominator *= (1-psw)
        # P(spam|word1word2…wordn) = P1P2…Pn/(P1P2…Pn+(1-P1)(1-P2)…(1-Pn))
        resProb = numerator/(numerator+denominator)
        if resProb > 0.9:
            result.append(1)
        else:
            result.append(0)
    
    #计算准确率、精确度和召回率
    rightCnt = 0 
    TP = 0  #将正类预测为正类数 
    FN = 0  #将正类预测为负类数
    FP = 0  #将负类预测为正类数
    for index in range(len(testLabel)):
        if testLabel[index] == 1:
            if result[index] == 1:
                rightCnt += 1
                TP += 1
            else:
                FN +=1
        else:
            if result[index] == 0:
                rightCnt += 1
            else:
                FP +=1
    accuracy = rightCnt / len(testLabel)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return accuracy,precision,recall

def main():
    allEmail,label = getFileDiv()
    trainEmail = allEmail[:40]
    trainLable = label[:40]
    testEmail = allEmail[40:]
    testLabel = label[40:]
    hamDict,spamDict = getWordsProb(trainEmail, trainLable)
    accuracy,precision,recall = mulNBTest(hamDict,spamDict,testEmail,testLabel)
    print("%f%f%f" %(accuracy,precision,recall))
    return accuracy,precision,recall

if __name__ == '__main__':
    accuracy = []
    precision = []
    recall = []
    for i in range(100):
        a,b,c = main()
        accuracy.append(a)
        precision.append(b)
        recall.append(c)
    x = list(range(100))
    plt.plot(x, accuracy, color='red', label='准确率')
    plt.plot(x, precision, color='skyblue', label='精确度')
    plt.plot(x, recall, color='blue', label='召回率')
    plt.legend(loc = 'upper right')
    plt.show()