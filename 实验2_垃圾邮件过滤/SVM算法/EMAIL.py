import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection,metrics, svm,naive_bayes
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("ignore")

path = '..\\spam.csv'
data = pd.read_csv(path, encoding='latin-1')
count_Class = pd.value_counts(data["v1"], sort= True) #统计V1处的值ham和spam的数量
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()#显示处理数据中spam和ham的个数
"""count1 = Counter(" ".join(data[data['v1'] == 'ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
print(df1) 
count2 = Counter(" ".join(data[data['v1'] == 'spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
df1.plot.bar(legend = False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()
df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()"""#这一段都是用来计算未经处理的ham和spam中的特征值，根据需要可取消注释，无影响
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["v2"])
print(np.shape(X))#得到处理后的特征值
data["v1"]=data["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
print([np.shape(X_train), np.shape(X_test)])#将变量spam/non-spam转换为二进制变量，然后将数据集分为训练集和测试集

list_alpha = np.arange(1/100000, 20, 0.11) #样本容量
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0 #参数调入
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1
    #具体算法
matrix = np.asmatrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=10))#将数据存入dataframe中
best_index = models['Test Precision'].idxmax()
print(models.iloc[best_index, :])
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print(pd.DataFrame(data = m_confusion_test, columns = ['ham', 'spam'],index = ['ham', 'spam']))
accuracy = []
precision = []
recall = []
accuracy = models['Test Accuracy']
precision = models['Test Precision']
recall = models['Test Recall']
plt.plot( accuracy, color='red', label='accuracy')
plt.plot( precision, color='skyblue', label='precision')
plt.plot( recall, color='blue', label='recall')
plt.legend(loc='upper right')
my_x_ticks = np.arange(0, 9, 1) #X轴的范围和刻度值
plt.xticks(my_x_ticks)
plt.xlim((0, 9))#X轴那个线的范围
plt.show()
#上面这一坨为朴素贝叶斯算法
list_C = np.arange(500, 2000, 100) #100000 #样本容量 （500-2000，以100为间隔分组）
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))
recall_test = np.zeros(len(list_C))
precision_test= np.zeros(len(list_C))
count = 0
#参数调入
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
    count = count + 1
    #具体算法
matrix = np.asmatrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns =
             ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
print(models.head(n=10))
#数据转为dataframe的形式
best_index = models['Test Precision'].idxmax()
print(models.iloc[best_index, :])

m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
print(pd.DataFrame(data = m_confusion_test, columns = ['ham', 'spam'],index = ['ham', 'spam']))
accuracy = []
precision = []
recall = []
accuracy = models['Test Accuracy']
precision = models['Test Precision']
recall = models['Test Recall']
plt.plot( accuracy, color='red', label='accuracy')
plt.plot( precision, color='skyblue', label='precision')
plt.plot( recall, color='blue', label='recall')
plt.legend(loc='upper right')
plt.xlim((0, 9))#调X轴范围
my_x_ticks = np.arange(0, 9, 1)#调刻度
plt.xticks(my_x_ticks)
plt.show()
#svm算法