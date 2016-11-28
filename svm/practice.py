"""
Stripped-down version of the face recognition example by Olivier Grisel

http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html

## original shape of images: 50, 37
"""
import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
from sklearn.datasets import load_sample_image, load_sample_images
from scipy.misc import imresize
import logging
from os.path import dirname, join
from sklearn.externals import joblib

logging.basicConfig()
# ..
# .. load data ..
# lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# print(lfw_people.data.shape)
data = []
target = np.genfromtxt('train.csv', delimiter=',')[:10,1][1:]
for i in range(1,len(target)+1):
    num_zeroes = (5 - len(str(i))) * '0'
    l_img = load_sample_image(num_zeroes+str(i)+'.jpg')
    # l_img = imresize(l_img, (8,8,3))
    data.append(l_img)

lfw_people = datasets.base.Bunch(target=np.array(target), data=np.array(data).reshape(len(data), -1))

print(lfw_people.data.shape)
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
skf = model_selection.StratifiedKFold(n_splits=4)
train, test = next(iter(skf.split(lfw_people.data, lfw_people.target)))
X_train, X_test = faces[train], faces[test]
y_train, y_test = lfw_people.target[train], lfw_people.target[test]

# ..
# .. dimension reduction ..
pca = decomposition.PCA(svd_solver='randomized', n_components=150, whiten=True)

X_train_pca = pca.fit_transform(X_train)
# X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(X_train_pca.shape, X_test_pca.shape)
# ..
# .. classification ..
clf = svm.SVC(C=5., gamma=0.001)
# clf.fit(X_train_pca, y_train)
clf.fit(X_train, y_train)


# saving model to file
filename = raw_input("Finished Training. Please enter file name (eg. abc.pkl) to save model: ")
joblib.dump(clf, filename) 



print 'Score on unseen data: '
# print clf.score(X_test_pca, y_test)
print clf.score(X_test, y_test)
