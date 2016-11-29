"""
Stripped-down version of the face recognition example by Olivier Grisel

http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html

## original shape of images: 50, 37
"""
import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
from sklearn.datasets import load_sample_image, load_sample_images
from sklearn.preprocessing import scale
from scipy.misc import imresize
import logging
from os.path import dirname, join
from sklearn.externals import joblib
import sys
import util

logging.basicConfig()
# ..
# .. load data ..
# lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# print(lfw_people.data.shape)

NUM_SAMPLES = 7000
NUM_COMP_PCA = 250
print "loading data..."
'''
data = []
target = np.genfromtxt('train.csv', delimiter=',')[:NUM_SAMPLES,1][1:]
for i in range(1,len(target)+1):
    num_zeroes = (5 - len(str(i))) * '0'
    l_img = load_sample_image(num_zeroes+str(i)+'.jpg')
    # l_img = imresize(l_img, (8,8,3))
    data.append(l_img)
    sys.stdout.write(".")
    sys.stdout.flush()
'''
data, target = util.load_data('../train', 'train.csv', NUM_SAMPLES)
X_Y = datasets.base.Bunch(target=np.array(target), data=np.array(data).reshape(len(data), -1))

#X = np.reshape(lfw_people.data, (lfw_people.data.shape[0], -1))
X = X_Y.data
Y = X_Y.target
print(X.shape, Y.shape)
print "separating train and test data..."
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.10, random_state=17)
#skf = model_selection.StratifiedKFold(n_splits=4)
#train, test = next(iter(skf.split(lfw_people.data, lfw_people.target)))
#X_train, X_test = faces[train], faces[test]
#y_train, y_test = lfw_people.target[train], lfw_people.target[test]
#print y_train, y_test

# ..
# .. dimension reduction ..
print "doing pca..."
pca = decomposition.PCA(svd_solver='randomized', n_components=NUM_COMP_PCA, whiten=True)
pca.fit(X_train)
X_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)
print X_pca.shape

# ..
# .. classification ..
#clf = svm.SVC(C=5., gamma=0.001)
print "simple SVC..."
clf = svm.SVC()
# clf.fit(X_train_pca, y_train)
clf.fit(X_pca, Y_train)
print "training set score"
print clf.score(X_pca, Y_train)
print "test set score"
print clf.score(X_test_pca, Y_test)


print "simple rbf SVC..."
clf = svm.SVC(kernel='rbf')
# clf.fit(X_train_pca, y_train)
clf.fit(X_pca, Y_train)
print "training set score"
print clf.score(X_pca, Y_train)
print "test set score"
print clf.score(X_test_pca, Y_test)

print "rbf small C=5 SVC..."
clf = svm.SVC(kernel='rbf', C=5, gamma=0.001)
# clf.fit(X_train_pca, y_train)
clf.fit(X_pca, Y_train)
print "training set score"
print clf.score(X_pca, Y_train)
print "test set score"
print clf.score(X_test_pca, Y_test)


print "rbf large C=100 SVC..."
clf = svm.SVC(kernel='rbf', C=100, gamma=0.001)
# clf.fit(X_train_pca, y_train)
clf.fit(X_pca, Y_train)
print "training set score"
print clf.score(X_pca, Y_train)
print "test set score"
print clf.score(X_test_pca, Y_test)



print "no pca at all simple svc..."
clf = svm.SVC()
# clf.fit(X_train_pca, y_train)
clf.fit(X, Y)
print "training set score"
print clf.score(X, Y)
print "test set score"
print clf.score(X_test, Y_test)
# saving model to file
#filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
#joblib.dump(clf, filename) 



#print 'Score on unseen data: '
#print clf.score(X_test_pca, y_test)
#print clf.score(X_test, y_test)
