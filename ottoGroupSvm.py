from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from loaddatas import load_dataset
from sklearn import svm
from sklearn.externals import joblib
import numpy as np

X_train,Y_train,X_test,Y_test = load_dataset();

print('X_train.shape:'+str(X_train.shape))
print('Y_train.shape:'+str(Y_train.shape))
print('X_test.shape:'+str(X_test.shape))
print('Y_test.shape:'+str(Y_test.shape))
print('X_train.type:'+str(type(X_train)))
print('Y_train.type:'+str(type(Y_train)))

#SVM参数设置
n_classes = 9
print("Fiting the classifier to the training set")
t0 = time()
param_grid = {'c':[100000],'gamma':[0.00025,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.1],}

#开始进行SVM模型训练
#clf =  GridSearchCV(SVC(kernel='rbf'),param_grid)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr',probability=True)
clf = clf.fit(X_train,Y_train)
print("Done Fiting in %0.3fs"%(time()-t0))

joblib.dump(clf,"train_model.m")
print("Best estimotor found by grid search:")
t0 = time()

pred_label = clf.predict_proba(X_test)
joblib.dump(pred_label,"pred_label.m")
np.savetxt('submission.csv',pred_label,delimiter=',')
