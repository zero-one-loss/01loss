from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import time
import os
import sys
sys.path.append('..')
from core.svmboost import SVMB
import pickle
import argparse


parser = argparse.ArgumentParser(description='svm')

parser.add_argument('--c', default=0.01, type=float,
                    help='c')
parser.add_argument('--votes', default=100, type=int,
                    help='number of votes')
parser.add_argument('--ratio', default=0.66, type=float,
                    help='ratio')
parser.add_argument('--dual', action='store_true', help='Dual')
parser.add_argument('--target', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
args = parser.parse_args()


dual = True if args.dual else False

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


svc = SVMB(args.votes, args.c, args.ratio, dual)
a = time.time()
svc.fit(train, train_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = svc.predict(test)

print('Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=svc.predict(train)))
print('Test Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, args.target), 'wb') as f:
    pickle.dump(svc, f)

# index = np.arange(test_label.shape[0])[yp == test_label]
# np.save('svm_01_index.npy', index)