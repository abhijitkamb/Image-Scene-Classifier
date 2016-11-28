"""
Stripped-down version of the face recognition example by Olivier Grisel

http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html

## original shape of images: 50, 37
"""

import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
import logging
logging.basicConfig()
# ..
# .. load data ..
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print type(lfw_people)
#lfw_people = load_sample_images()
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
skf = model_selection.StratifiedKFold(n_splits=4)
train, test = next(iter(skf.split(lfw_people.data, lfw_people.target)))
X_train, X_test = faces[train], faces[test]
y_train, y_test = lfw_people.target[train], lfw_people.target[test]

# ..
# .. dimension reduction ..
pca = decomposition.PCA(svd_solver='randomized', n_components=150, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# ..
# .. classification ..
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

print 'Score on unseen data: '
print clf.score(X_test_pca, y_test)



