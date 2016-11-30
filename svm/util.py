from PIL import Image
from scipy import misc
from sklearn import cluster
import numpy as np
import glob

def load_data(x_dir, y_csv_file, size, test):

    prefix = ''
    if size == 1:
        prefix = '00001'
    elif size == 10:
        prefix = '0000'
    elif size == 100:
        prefix = '000'
    elif size == 1000:
        prefix = '00'

    if test == 1:
        prefix = 'test_' + prefix 

    X_train_files = glob.glob(x_dir + '/' + prefix + '*.jpg')
    X_train_files.sort()
    X_train = np.array([np.array(Image.open(fname)) for fname in X_train_files])

    # this is an (N, 1) array; remember to convert to 1-hot for CNN
    y_train = np.genfromtxt(y_csv_file, delimiter=",")[:size+1,1][1:]
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train

def compressImg(x_dir, size=7000, test=0):
    prefix = ''
    if size == 1:
        prefix = '00001'
    elif size == 10:
        prefix = '0000'
    elif size == 100:
        prefix = '000'
    elif size == 1000:
        prefix = '00'

    if test == 1:
        prefix = 'test_' + prefix

    X_train_files = glob.glob(x_dir + '/' + prefix + '*.jpg')
    X_train_files.sort()

    for fname in X_train_files:
        data = np.array(Image.open(fname))
        data = np.array(data).reshape(-1, 1)

        k_means = cluster.KMeans(n_clusters=5)
        k_means.fit(data)

        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        data_compressed = np.choose(labels, values)
        data_compressed.shape = data.shape
        misc.imsave('compressed/'+ fname.split("/")[1], data_compressed.reshape(128, 128, 3))