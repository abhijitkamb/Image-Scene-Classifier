import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
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
from skimage import data, color, exposure
#def load_data(x_dir, y_csv_file, size, test):
from skimage.feature import hog

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

size = 7000

X_train_files = glob.glob('compressed/*.jpg')
X_train_files.sort()
X = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_train_files[:size-1]])

Y = np.genfromtxt('train.csv', delimiter=",")[:size,1][1:]


#for i in range(1,9):
#    print(np.sum(y_train == i))

print(X.shape, Y.shape)
#print(X_train_files[:size])
#print(Y)

Xhog = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), transform_sqrt=True)) for x in X])

print(Xhog.shape, Y.shape)

#Xhog = StandardScaler().fit_transform(Xhog)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Xhog.reshape(len(Xhog), -1), Y, test_size=0.10, random_state=37)

clf1 = svm.SVC(kernel='rbf', C=1, gamma=0.01, probability=True, cache_size=1000)

clf1.fit(X_train, Y_train)

print "clf1 training set score"
print clf1.score(X_train, Y_train)
print "clf1 test set score"
print clf1.score(X_test, Y_test)

p = clf1.predict(X_test)
print(p[:21], Y_test[:21])

d_pred= {}
d_test = {}
for i in range(1, 9):
    d_pred[i] = 0
    d_test[i] = np.sum(Y_test == i)
    for idx, pred in enumerate(p):
        if pred == i and Y_test[idx] == i:
            if i in d_pred:
                d_pred[i] += 1


print "predicted distribution: "
print d_pred
print "actual distribution: "
print d_test

filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
joblib.dump(clf1, filename)
