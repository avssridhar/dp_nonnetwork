
import math
import numpy as np
from sklearn.base import BaseEstimator
import random

class Node(BaseEstimator):
    def __init__(self, feature_idx, threshold, left_tree, right_tree):
        self.feature_idx = feature_idx
        self.threshold = threshold #阈值
        self.left_tree = left_tree #左子树
        self.right_tree = right_tree #右子树
        
        
    def fillPredict(self, X, outputs, index):
#        print self.feature_idx
        split = X[:, self.feature_idx] < self.threshold
        left_index = index & split
        right_index = index & ~split
        
       
        if self.left_tree is not None: 
            self.left_tree.fillPredict(X, outputs, left_index)
        if self.right_tree is not None:
            self.right_tree.fillPredict(X, outputs, right_index)





def midPoints(x):
    '''
    eg.x=array([1,2,3,4])
    return array([1.5,2.5,3.5])
    '''
    return (x[1:] + x[:-1]) / 2.0


def majority(y, classes):
    if classes is None:
        classes = np.unique(y)  
    votes = np.zeros(len(classes))  
    for i, c in enumerate(classes):
        votes[i] = np.sum(y == c)  
    majority_idx = np.argmax(votes)
#    print int(votes[majority_idx])
    return classes[majority_idx]




def Gini(classes, y, sample_weight):
    sum_squares = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    else:
        n2 = float(n*n)
        if sample_weight is not None:
            for c in classes: 
                count = np.sum(y == c)
                getindex = np.where(y == c)[0]
                w = np.sum(sample_weight[getindex])
                # count -= w * count
                c2 = ((count * w) ** 2) / n2
                sum_squares += c2
        else:
            for c in classes:  
                count = np.sum(y == c)
                c2 = (count ** 2) / n2
                sum_squares += c2
        return 1 - sum_squares



def findBestGiniSplit(classes, col_vector, thresholds, y, sample_weight):
    best_score = 999999999
    best_thresh = None
    n = len(y)

    
    for t_index in thresholds:
        index = col_vector < t_index  
        left_labels = y[index]  
        right_labels = y[~index] 
        left_score = Gini(classes, left_labels, sample_weight)  # Gini(D1)
        right_score = Gini(classes, right_labels, sample_weight)  # Gini(D2)
        left_n = len(left_labels)  
        right_n = len(right_labels) 
        # Gini(D,A) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        totalScore = (left_n / n) * left_score + (right_n / n) * right_score
        if totalScore < best_score:
            best_score = totalScore
            best_thresh = t_index
    return best_thresh, best_score


def loadDataSet(fileName):
    n_features = len(open(fileName).readline().split(','))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArray = []
        curLine = line.strip().split(',')
        for i in range(n_features - 1):
            lineArray.append(float(curLine[i]))
        dataMat.append(lineArray)
        labelMat.append(float(curLine[-1]))
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
#    map(lambda label: 0 if label==0 else 1, labelMat)
    return dataMat, labelMat


def splitData(X, y, trainPre=0.8):
    n_rows, n_cols = X.shape
    ind = np.arange(n_rows)
#    np.random.shuffle(ind)
    n_train = int(n_rows * trainPre)
    train_ind = ind[:n_train]
    test_ind = ind[n_train:]
    X_train = X[train_ind, :]
    X_test = X[test_ind, :]
    y_train = y[train_ind]
    y_test = y[test_ind]
    return X_train, y_train, X_test, y_test


def horizontallySplitData(X, y, section=5):
    if section == 1:
        return X, y
    else:
        n_rows = X.shape[0]
        rangelist = range(1, n_rows)
        random.seed(12345)
        valuelist = random.sample(rangelist, section - 1)
        valuelist.sort()
        valuelist.append(n_rows)
#       print "valuelist", valuelist
        k = 0
        j = 0
        h_X = {}
        h_y = {}
        for i in valuelist:
            assert j <= section - 1
            h_X[j] = X[k:i]
            h_y[j] = y[k:i]
            k = i
            j += 1
        # print h_X, h_y
        return h_X, h_y


def RMSE(y, yhat):
    return np.sqrt(np.mean((yhat - y)**2))


def scoreAcc(y, yhat):
    n = len(y)
    tp = np.sum(yhat * y)
    tn = np.sum((1 - yhat) * (1 - y))
    acc = float((tp + tn) / n)
    return acc
    # return float(np.sum(yhat == y)) / len(y)


import scipy as sp


def logLoss(y, yhat):
    epsilon = 1e-15
    yhat = sp.maximum(epsilon, yhat)
    yhat = sp.minimum(1 - epsilon, yhat)
    ll = sum(y * sp.log(yhat) + sp.subtract(1, y)
             * sp.log(sp.subtract(1, yhat)))
    ll = -1.0 / len(y) * ll
    return ll


def weightScore(w):
    F = []
    for w_p in w:
        F_p = w_p / np.sum(w)
        F.append(F_p)
    return F

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def laplaceMechanism(privacy):

#    if privacy is not None:
#        value = (privacy / 2.0) * math.exp(-1 * privacy * math.fabs(x))
#        return value
#    else:
#        return 0.0

#     mu = 0
#     b = 1.0 / privacy
#     a = random.uniform(-0.5, 0.5)
#     return mu - b * sgn(a) * math.log(1 - 2 * math.fabs(a))

    return 0.0



def AdaboostPL(num_learners, section, Ada_set_p, alpha_p, lamb, X_test):
    #    print "Ada_set_p", Ada_set_p
    #    print "alpha_p", alpha_p
    #    print "lamb", lamb
    predTotal = np.zeros(X_test.shape[0], dtype=float)
    for t in range(num_learners[0]):
        pred_t = np.zeros(X_test.shape[0], dtype=float)
        alpha_t = 0.0
        for id in range(section):
            #            print "Ada_set_p[id]", Ada_set_p[id]
            #            print "Ada_set_p[id][t]", Ada_set_p[id][t]
            temp = Ada_set_p[id][t].predict(X_test)
            temp[temp == 0] = -1
            pred_t += temp
            alpha_t += alpha_p[id][t] * lamb[id]
#        print "alpha_t", alpha_t
#        print "pred_t", pred_t
#        pred_t = [ 0 if i==0 else 1 for i in pred_t]
        predTotal += pred_t * alpha_t

    predTotal = [0 if j <= 0 else 1 for j in predTotal]
#    print "predTotal", predTotal
    return predTotal

class Leaf(object):
    def __init__(self, v):
        self.v = v
        
    def toStr(self, split="#"):
        return split + "<" + str(self.v) + ">"
    
    def __str__(self):
        return self.toStr
    
    def predict(self, X):
        X = np.atleast_2d(X)
        output = np.zeros(X.shape[0])
        output[:] = self.v
        return output
        
    def fillPredict(self, X, output, index):
        output[index] = self.v

class DecisionTree(BaseEstimator):
    def __init__(self,\
                 gt_privacy_p,\
                 min_samples_leaf=1,\
                 max_depth=6):
        self.root = None 
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.gt_privacy_p = gt_privacy_p
        self.classes = None
        self.getThresholds = self.thresholds
        self.logger_string='Logging'

    def thresholds(self, x):
        if len(x) > 1:
            return midPoints(np.unique(x))
        else:
            return x

    def splitData(self, X, y, h,sample_weight, epsilon):
        n_rows, n_features = X.shape
        if n_rows <= self.min_samples_leaf or h > self.max_depth:
            self.n_leaf += 1  
            leaf = majority(y, self.classes)  
            if epsilon > 0.0:
                leaf += math.floor(laplaceMechanism(epsilon))
            return Leaf(leaf)
        
        elif np.all(y == y[0]): 
            self.n_leaf += 1  
            return Leaf(y[0])  
            
        else:
            best_split_score = 999999999
            best_feature_idx = None
            best_threshold = None
            for feature_t in range(n_features):
                col_vector = X[:, feature_t] 
                feature_vector = self.getThresholds(col_vector)
                thresh, totalScore = findBestGiniSplit(
                    self.classes, col_vector, feature_vector, y, sample_weight)
                if thresh is not None:
                    if totalScore < best_split_score:  
                        best_split_score = totalScore
                        best_feature_idx = feature_t  
                        best_threshold = thresh  
                else:
                    break

            if best_feature_idx is not None:
#              
                if epsilon > 0.0:
                    best_threshold += math.floor(laplaceMechanism(0.5*epsilon))
                
                
                left_branch = X[:, best_feature_idx] < best_threshold
                right_branch = ~left_branch
                self.logger_string += f"lb.len: {len(left_branch)}\n"
                self.logger_string += f"rb.len: {len(right_branch)}\n"


                left_data = X[left_branch, :]  
                right_data = X[right_branch, :]  
                self.logger_string += f"ld.size: {left_data.shape}\n"
                self.logger_string += f"rd.size: {right_data.shape}\n"
                left_labels = y[left_branch]  
                right_labels = y[right_branch]  

                
                epsilon_left = epsilon_right = 0
                if left_data.shape[0] == 0 and right_data.shape[0] != 0:
                    epsilon_right = 0.5 * epsilon
                elif left_data.shape[0] != 0 and right_data.shape[0] == 0:
                    epsilon_left = 0.5 * epsilon
                else:
                    epsilon_left = 0.25 * epsilon
                    epsilon_right = 0.25 * epsilon

                
                del y
                del X
                del left_branch
                del right_branch

                
                left_tree,left_string= self.splitData(
                           left_data, left_labels, h+1,sample_weight, epsilon_left)
                right_tree,right_string = self.splitData(
                           right_data, right_labels, h+1,sample_weight, epsilon_right)
               
                node = Node(best_feature_idx, best_threshold,\
                            left_tree, right_tree)
                return node

    def fit(self, X, y,sample_weight=None):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        self.classes = np.unique(y)  
        self.n_classes = len(self.classes)  
        self.n_leaf = 0  

        self.root= self.splitData(X, y, 1, sample_weight, self.gt_privacy_p)
        

    def predict(self, X):
        X = np.atleast_2d(X)
        n_rows = X.shape[0]

        
        outputs = np.zeros(n_rows)
        index = np.ones(n_rows, dtype='bool')
        if self.root is not None:
            self.root.fillPredict(X, outputs, index)
        return outputs
    
    def return_logger(self,str):
        print(self.logger_string)
        #return self.logger_string
