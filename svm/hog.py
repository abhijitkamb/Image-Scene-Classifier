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

size = 3001

X_train_files = glob.glob('../train/*.jpg')
X_train_files.sort()
X = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_train_files[:size-1]])

Y = np.genfromtxt('train.csv', delimiter=",")[:size,1][1:]


#for i in range(1,9):
#    print(np.sum(y_train == i))

print(X.shape, Y.shape)
#print(X_train_files[:size])
#print(Y)

Xhog = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), transform_sqrt=True)) for x in X])

print(Xhog.shape, Y.shape)

#Xhog = StandardScaler().fit_transform(Xhog)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Xhog.reshape(len(Xhog), -1), Y, test_size=0.10, random_state=37)


clf = svm.SVC(kernel='rbf', C=1, gamma=0.01)
clf.fit(X_train, Y_train)
print "training set score"
print clf.score(X_train, Y_train)
print "test set score"
print clf.score(X_test, Y_test)

p = clf.predict(X_test)
print(p[:21], Y_test[:21])

for i in range(1, 9):
    print(np.sum(Y_test == i), np.sum(p == i))

filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
joblib.dump(clf, filename)
