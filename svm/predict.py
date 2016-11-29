from sklearn.externals import joblib
from sklearn import model_selection, datasets, decomposition, svm
import numpy as np
from sklearn.datasets import load_sample_image, load_sample_images
import sys
import util

filename = raw_input("Enter file name (eg. abc.pkl) of the saved model: ")
clf = joblib.load(filename)


print 'loading data...'

data = []
NUMSAMPLES = 1000
ISTEST = 1

'''
for i in range(1, testsize+1):
    num_zeroes = (5 - len(str(i))) * '0'
    l_img = load_sample_image('test_'+num_zeroes+str(i)+'.jpg')
    data.append(l_img)
    sys.stdout.write('.')
'''

data, target = util.load_data('../val', 'train.csv', NUMSAMPLES, ISTEST)


print 'predicting...'
test = np.array(data).reshape(len(data), -1)
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

