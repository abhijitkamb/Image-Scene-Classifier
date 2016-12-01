import numpy as np
from sklearn import model_selection, datasets, decomposition, svm
from sklearn.externals import joblib
from PIL import Image
import glob
from skimage import data, color, exposure, filters
from skimage.feature import hog
import cv2

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


size = 100

X_train_files = glob.glob('/mnt/c/Users/akila/PycharmProjects/CSC411_A3/compressed/*.jpg')
X_train_files.sort()
X = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_train_files[:size-1]])

Y = np.genfromtxt('train.csv', delimiter=",")[:size,1][1:]

#for i in range(1,9):
#    print(np.sum(y_train == i))

print("Init params", X.shape, Y.shape)
#print(X_train_files[:size])
#print(Y)

# Set up Hog1
Xhog1 = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2), transform_sqrt=True))\
                   for x in X])
# Set up Hog2
hog2 = cv2.HOGDescriptor()
hog2.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
Xhog2 = np.array([hog2.detectMultiScale(x, winStride=(4, 4), padding=(8, 8), scale=1.05) for x in X])

print("Hog params: ", Xhog1.shape, Y.shape)

#Xhog = StandardScaler().fit_transform(Xhog)

X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(Xhog1.reshape(len(Xhog1), -1), Y, test_size=0.10, random_state=37)

X_train2, X_test2, Y_train2, Y_test2 = \
    model_selection.train_test_split(Xhog2.reshape(len(Xhog2), -1), Y, test_size=0.10, random_state=37)

clf1 = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True, cache_size=1000)
clf2 = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True, cache_size=1000)
clf = VotingClassifier(estimators=[('hog', clf1), ('sobel', clf2)], voting='hard', weights=[1,1])

clf1.fit(X_train, Y_train)
clf2.fit(X_train2, Y_train2)
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

# def pedestrianDetector(args):
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     # loop over the image paths
#     for imagePath in paths.list_images(args["images"]):
#         # load the image and resize it to (1) reduce detection time
#         # and (2) improve detection accuracy
#         image = cv2.imread(imagePath)
#         image = imutils.resize(image, width=min(400, image.shape[1]))
#         orig = image.copy()
#
#         # detect people in the image
#         (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
#                                                 padding=(8, 8), scale=1.05)
#
#         # draw the original bounding boxes
#         for (x, y, w, h) in rects:
#             cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         # apply non-maxima suppression to the bounding boxes using a
#         # fairly large overlap threshold to try to maintain overlapping
#         # boxes that are still people
#         rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#         pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
#
#         # draw the final bounding boxes
#         for (xA, yA, xB, yB) in pick:
#             cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
#
#         # show some information on the number of bounding boxes
#         filename = imagePath[imagePath.rfind("/") + 1:]
#         print("[INFO] {}: {} original boxes, {} after suppression".format(
#             filename, len(rects), len(pick)))
#
#         # show the output images
#         cv2.imshow("Before NMS", orig)
#         cv2.imshow("After NMS", image)
#         cv2.waitKey(0)