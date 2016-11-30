import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_sample_image, load_sample_images
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from scipy.misc import imresize
import logging
from os.path import dirname, join
from sklearn.externals import joblib
import sys
import util
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
from sklearn import cluster
import glob
from skimage import data, color, exposure, filters
#def load_data(x_dir, y_csv_file, size, test):
from skimage.feature import hog
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

def getDataDistribution(Y_test, p, name):
    d_pred = {}
    d_test = {}
    for i in range(1, 9):
        d_pred[i] = 0
        d_test[i] = np.sum(Y_test == i)

        for idx, pred in enumerate(p):
            if pred == i and Y_test[idx] == i:
                if i in d_pred:
                    d_pred[i] += 1

    print name + " predicted distribution: "
    print d_pred
    print name + " actual distribution: "
    print d_test

    return d_pred, d_test


size = 9000

X_train_files = glob.glob('compressed/*.jpg')
X_train_files.sort()
X = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_train_files[:size-1]])

Y = np.genfromtxt('train.csv', delimiter=",")[:size,1][1:]

#for i in range(1,9):
#    print(np.sum(y_train == i))

print("Init params", X.shape, Y.shape)
#print(X_train_files[:size])
#print(Y)

Xhog = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True)) for x in X])

print("Hog params: ", Xhog.shape, Y.shape)

#Xhog = StandardScaler().fit_transform(Xhog)

X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(Xhog.reshape(len(Xhog), -1), Y, test_size=0.10, random_state=37)

clf1 = svm.SVC(kernel='rbf', C=10, gamma=1, probability=True, cache_size=1000)
clf2 = svm.SVC(kernel='rbf', C=1, gamma=0.01, probability=True, cache_size=1000)
clf = VotingClassifier(estimators=[('hog', clf1), ('sobel', clf2)], voting='hard', weights=[1,1])

clf1.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)
clf.fit(X_train, Y_train)

print "clf1 training set score"
print clf1.score(X_train, Y_train)
print "clf1 test set score"
print clf1.score(X_test, Y_test)

p1 = clf1.predict(X_test)
print(p1[:21], Y_test[:21])

getDataDistribution(Y_test, p1, 'clf1')

print "clf2 training set score"
print clf2.score(X_train, Y_train)
print "clf2 test set score"
print clf2.score(X_test, Y_test)

p2 = clf2.predict(X_test)
print(p2[:21], Y_test[:21])

getDataDistribution(Y_test, p2, 'clf2')

print "clf training set score"
print clf.score(X_train, Y_train)
print "clf test set score"
print clf.score(X_test, Y_test)

p = clf.predict(X_test)
print(p[:21], Y_test[:21])

getDataDistribution(Y_test, p, 'clf')

filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
joblib.dump(clf, filename)
