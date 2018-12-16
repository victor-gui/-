# -*- coding:utf-8 -*-
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import ensemble,feature_extraction,preprocessing
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


train_data = pd.read_csv(open("train.csv"))
test_data = pd.read_csv(open("test.csv"))

x_label = []
for i in range(1,94):
    x_label.append("feat_%s"%(i))
train_x = np.array(train_data[x_label])
test_x = np.array(test_data[x_label])


trainy = train_data["target"]
trainlabel = np.zeros(len(trainy))

for i in range(len(trainy)):
    trainlabel[i] = int(trainy[i][-1])
train_y = np.array(trainlabel, dtype="int")


print(train_x.shape,train_y.shape,test_x.shape)  # (49502, 93) (49502, 9) (12376, 93)


def print_best_score(gsearch, param_test):
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


lensesLabel = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]

clf = RandomForestClassifier(n_estimators=400,n_jobs=-1)
clf = clf.fit(train_x,train_y)

test_y = clf.predict_proba(test_x)

answer = pd.read_csv(open("sampleSubmission.csv"))
class_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]

# 将答案放进去
j = 0
for class_name in class_list:
    answer[class_name] = test_y[:,j]
    # answer[test_y[:, j]] = 1
    j += 1
answer.to_csv("submission_tree.csv",index=False)  # 不要保存引索列






















