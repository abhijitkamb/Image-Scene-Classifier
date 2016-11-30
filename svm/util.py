from PIL import Image
import numpy as np
import glob

def load_data(x_dir, y_csv_file, size, test):

    prefix = ''
    if size == 10:
        prefix = '0000'
    elif size == 100:
        prefix = '000'
    elif size == 1000:
        prefix = '00'

    if test == 1:
        prefix = 'test_' + prefix 

    X_train_files = glob.glob(x_dir + '/' + prefix + '*.jpg')
    X_train_files.sort()
    print X_train_files
    X_train = np.array([np.array(Image.open(fname)) for fname in X_train_files])

    # this is an (N, 1) array; remember to convert to 1-hot for CNN
    y_train = np.genfromtxt(y_csv_file, delimiter=",")[:size,1][1:]
    for x,y in zip(X_train_files, y_train):
        print (x, y)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train
