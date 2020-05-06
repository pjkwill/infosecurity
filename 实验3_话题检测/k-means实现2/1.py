
import os
import jieba 
from jieba.analyse import *
import numpy
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib


def trainfenci():
	trainpath = '..\\话题检测\\训练集\\'
	resultpath = '..\\话题检测\\train.txt'

	i = 1
	while i<=22:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C4-Literature\\' + 'Literature ('+str(i)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data,topK = 10, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		i = i+1
	with open (resultpath,'a') as rf:
		rf.write('\n')

	j = 1
	while j<=34:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C5-Education\\' + 'Education ('+str(j)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 7, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		j = j+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')

	k = 1
	while k<=152:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C7-History\\' + 'History ('+str(k)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		k = k+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')
	l = 1
	while l<=13:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C17-Communication\\' + 'Communication ('+str(l)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 5, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		l = l+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')
	m = 1
	while m<=59:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C34-Economy\\' + 'Economy ('+str(m)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		m = m+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')

	n = 1
	while n<=100:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C39-Sports\\' + 'Sports ('+str(n)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
		n = n+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')


def jieba_tokenize(text):
    return jieba.lcut(text) 

tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=False)
'''
tokenizer: 指定分词函数
lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
所以最好是False
'''
def train():
	trainpath = '..\\话题检测\\train.txt'
    # file 文件类型的对象
	with open(trainpath,'r') as file:
		# 以列表的形式输出文本

		trainlines = list(file)
		#print(lines)
	train_list = trainlines
	#print(text_list)
    #需要进行聚类的文本集
	train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_list)
	num_clusters = 6
	km_cluster = KMeans(n_clusters=num_clusters, max_iter=999999, n_init=1, init='k-means++',n_jobs=1)
	trainresult = km_cluster.fit_predict(train_tfidf_matrix)
	print ("Train result: ", trainresult)
	print ('\n')
	print ("cluster_center: ")
	print (km_cluster.cluster_centers_)
	print ('\n')
	i = 1
	'''
	j=1
	label = {0:"Literature", 1:"Education", 2:"History", 3:"Communication", 4:"Economy", 5:"Sports"}
	train_result_path = '..\\话题检测\\训练集结果.txt'
	with open(train_result_path,'w') as file:
		while j <= len(km_cluster.labels_):
			if j<10:
				file.write (str(j)+'           '+str(km_cluster.labels_[j - 1])+'    '+label[km_cluster.labels_[j - 1]]+'\n')
			
			else:
				file.write (str(j)+'          '+str(km_cluster.labels_[j - 1])+'    '+label[km_cluster.labels_[j - 1]]+'\n')
			j = j + 1
	'''
	while i <= len(km_cluster.labels_):
		if i<10:
			print (i,'           ',km_cluster.labels_[i - 1])

		else:
			print (i,'          ',km_cluster.labels_[i - 1])
		i = i + 1
	print ('\n')
	print ("inertia:")
	print (km_cluster.inertia_)
	print ('\n')
	joblib.dump(km_cluster,'km.pkl')    #保存模型

def testfenci():
	trainpath = '..\\话题检测\\测试集\\'
	resultpath = '..\\话题检测\\test.txt'

	i = 1
	while i<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C4-Literature\\' + 'Literature ('+str(i)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data,topK = 10, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		i = i+1
	with open (resultpath,'a') as rf:
		rf.write('\n')

	j = 1
	while j<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C5-Education\\' + 'Education ('+str(j)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 7, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		j = j+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')

	k = 1
	while k<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C7-History\\' + 'History ('+str(k)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		k = k+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')
	l = 1
	while l<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C17-Communication\\' + 'Communication ('+str(l)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 5, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		l = l+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')
	m = 1
	while m<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C34-Economy\\' + 'Economy ('+str(m)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		m = m+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')

	n = 1
	while n<=10:
		with open(resultpath,'a') as rf:
			path = trainpath + 'C39-Sports\\' + 'Sports ('+str(n)+').txt' 
			with open(path, 'r') as f:
				data = f.read()
			for keyword, weight in extract_tags(data, topK = 1, withWeight = True):
				print('%s %s' % (keyword, weight))
				rf.write(keyword+' ')
			rf.write('\n')
		n = n+1
	with open (resultpath, 'a') as rf:
		rf.write('\n')

def test():
	testpath = '..\\话题检测\\test.txt' #测试集路径
	with open (testpath,'r') as file :
		testlines = list(file)
	test_list = testlines
	test_tfidf_matrix = tfidf_vectorizer.fit_transform(test_list)
	km_cluster = joblib.load('km.pkl')   #加载模型
	testresult = km_cluster.fit_predict(test_tfidf_matrix)
	print ("Test result: ",testresult)
	print ("cluster_center: ")
	print (km_cluster.cluster_centers_)
	print ('\n')
	i = 1
	'''
	j=1
	label = {0:"Literature", 1:"Education", 2:"History", 3:"Communication", 4:"Economy", 5:"Sports"}
	test_result_path = '..\\话题检测\\测试集结果.txt'
	with open(test_result_path,'w') as file:
		while j <= len(km_cluster.labels_):
			if j<10:
				file.write (str(j)+'           '+str(km_cluster.labels_[j - 1])+'    '+label[km_cluster.labels_[j - 1]]+'\n')
			
			else:
				file.write (str(j)+'          '+str(km_cluster.labels_[j - 1])+'    '+label[km_cluster.labels_[j - 1]]+'\n')
			j = j + 1
	'''

	while i <= len(km_cluster.labels_):
		if i<10:
			print (i,'           ',km_cluster.labels_[i - 1])

		else:
			print (i,'          ',km_cluster.labels_[i - 1])
		i = i + 1
	print ('\n')
	print ("inertia:")
	print (km_cluster.inertia_)
	print ('\n')

if __name__ == '__main__':
	trainfenci()
	train()
	testfenci()
	test()

'''
num_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
'''
#返回各自文本的所被分配到的类索引