from sklearn.externals import joblib
from sklearn import model_selection, datasets, decomposition, svm
import numpy as np
from sklearn.datasets import load_sample_image, load_sample_images
import sys
import util
from PIL import Image
from scipy import misc
from sklearn import cluster
import glob
from skimage import data, color, exposure
from skimage.feature import hog

filename = raw_input("Enter file name (eg. abc.pkl) of the saved model: ")
clf = joblib.load(filename)


print 'loading data...'

data = []
NUMSAMPLES = 2970
#ISTEST = 1

size1 = 970
X_test_files = glob.glob('../val/*.jpg')
X_test_files.sort()
X1 = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_test_files[:size1]])

size2 = 2000
X_test2_files = glob.glob('../test_128/*.jpg')
X_test2_files.sort()
X2 = np.array([color.rgb2gray(np.array(Image.open(fname))) for fname in X_test2_files[:size2]])

print X1.shape, X2.shape
X = np.concatenate([X1,X2])
print X.shape

Xhog = np.array([ np.array(hog(x, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2), transform_sqrt=True)) for x in X])

print 'predicting...'
test = np.array(Xhog).reshape(len(Xhog), -1)
Y_test = clf.predict(test)
print Y_test
print Y_test.shape

# writing to csv file for submission
print 'writing to csv file...'
id = np.arange(1, NUMSAMPLES+1, 1)
print id.shape, Y_test.shape

id_str = id.tolist()

#print Y_test
Y_test = Y_test.astype(int)
#print Y_test
Y_test_str = Y_test.tolist()
#print Y_test_str
id_str.insert(0, 'Id') 
Y_test_str.insert(0, 'Prediction')
#colname = np.array([['Id', 'Prediction']])
#vals = np.column_stack((id_str, Y_test_str))
#vals = np.concatenate((c, preds))
vals = list(zip(id_str, Y_test_str))

print vals
np.savetxt(filename.split('.')[0] + '.csv', vals, delimiter=",", fmt="%s") 

print 'done!'

