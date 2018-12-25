import csv
import random
import math
import operator
from sklearn import neighbors


#加载数据集，split是训练集和测试集分离
def loadDataset(fileName,split,trainingSet=[],testSet=[]):
    with open(fileName,"rt") as csvfile:
        lines = csv.reader(csvfile) #读取每一行数据
        dataset = list(lines) #将数据转化为list形式
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        
#计算未知样本与其他已知样本的距离,length表示维度
def euclideanDistance(instance1,instance2,length): 
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

#返回最近的K个样本
def getNeighbors(k,trainingSet,testInstance):
    distances =[]
    length = len(testInstance)-1 #测试实例的维度,维度是从0开始的
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet, length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors =[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#在k个距离里面，选择出类数最多的这一类
def getResponse(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1] #前k行的最后一列数据，也就是类别
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] =1
    sortedVotes = sorted(classVotes.iteritems(), key= operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#计算每次测试集测试后的精确度
def getAccuracy(testSet,predictions):
    correct =0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct +=1
    return (correct/float(len(testSet)))*100

def main():
    trainingSet =[]
    testSet=[]
    split = 0.67
    loadDataset(r'D:\Java\Javacode\kNN_20181225\src\irisdata.txt',split,trainingSet,testSet) 
    print('Trainset:' + repr(len(trainingSet)))
    print('test set:' + repr(len(testSet)))
    
    predictions=[]
    k=3
    for i in range(len(testSet)):
        neighbors = getNeighbors(k,trainingSet, testSet[i])
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted=' + repr(result)+ ',actual=' + repr(testSet[i][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('accuracy:' + repr(accuracy)+'%')

main()        
            
        
        