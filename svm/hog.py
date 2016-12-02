import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_sample_image, load_sample_images
from sklearn.preprocessing import scale, StandardScaler, normalize
from scipy.misc import imresize
import logging
from os.path import dirname, join
from sklearn.externals import joblib
import sys
import util
from PIL import Image
from scipy import misc
from sklearn import cluster
import glob
from skimage import data, color, exposure, filters
#def load_data(x_dir, y_csv_file, size, test):
from skimage.feature import hog
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt



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
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learning_curve2(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")lt.plot(train_sizes, train_scores_mean, 'o-', color="r" label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



size = 7000

X_train_files = glob.glob('../train/*.jpg')
X_train_files.sort()
X = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_train_files[:size]])

Y = np.genfromtxt('train.csv', delimiter=",")[:size+1,1][1:]

#for i in range(1,9):
#    print(np.sum(y_train == i))

print("Init params", X.shape, Y.shape)
#print(X_train_files[:size])
#print(Y)

Xhog = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2, 2), transform_sqrt=True)) for x in X])

Xhogreshaped = Xhog.reshape(len(Xhog), -1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Xhogreshaped, Y, test_size=0.10, random_state=37)

clf = svm.SVC(kernel='rbf', C=10, gamma=0.01)#, class_weight=class_weights)
clf.fit(X_train, Y_train)
print "clf training set score"
print clf.score(X_train, Y_train)
print "clf test set score"
print clf.score(X_test, Y_test)

p = clf.predict(X_test)
print(p[:21], Y_test[:21])

getDataDistribution(Y_test, p, 'clf')


# plots

title = "Learning Curve"

#cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
plot_learning_curve2(clf, title, X_train, Y_train)

plt.show()



filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
joblib.dump(clf, filename)
