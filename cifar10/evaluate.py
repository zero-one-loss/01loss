import numpy as np
import pickle
import sys
sys.path.append('..')
import time
import os
from sklearn.metrics import accuracy_score


import argparse

parser = argparse.ArgumentParser(description='evaluation on data set')
parser.add_argument('--source', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
args = parser.parse_args()

curdir = os.getcwd()
os.chdir('../data/cifar10')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')
train_data = np.load('train_image.npy').reshape((-1, 32*32*3)) / 255
test_data = np.load('test_image.npy').reshape((-1, 32*32*3)) / 255
os.chdir(curdir)

# pick first two classes
np.random.seed(2018)
train = train_data[train_label < 2]
test = test_data[test_label < 2]
train_label = train_label[train_label < 2]
test_label = test_label[test_label < 2]
print('training data size: ')
print(train.shape)
print('testing data size: ')
print(test.shape)


save_path = 'checkpoints'


with open(os.path.join(save_path, args.source), 'rb') as f:
    scd = pickle.load(f)

yp = scd.predict(test)

print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Vote Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train, kind='vote')))

print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
yp = scd.predict(test, kind='vote')
print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))