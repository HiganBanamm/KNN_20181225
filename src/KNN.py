from sklearn import neighbors
from sklearn import datasets

#创建一个分类器，名字叫knn
knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
#print(iris)

knn.fit(iris.data, iris.target) #iris.data是一个四维的矩阵，含有4种类别，iris.target是一列，表示这150朵花所在的类别
                                #这里使用默认的参数K
predictedLabel =knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLabel)