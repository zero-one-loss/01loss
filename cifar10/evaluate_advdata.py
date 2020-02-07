import numpy as np
import pickle
import sys
sys.path.append('..')
import time
import os
from sklearn.metrics import accuracy_score
import argparse

parser = argparse.ArgumentParser(description='SCD 01 loss')

parser.add_argument('--epsilon', default=1, type=float,
                    help='epsilon')
parser.add_argument('--type', default='scd', type=str,
                    help='scd or svm')
parser.add_argument('--source', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--target', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--predict', default='svm.pkl', type=str,
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
train_data = train_data[train_label < 2]
test_data = test_data[test_label < 2]
train_label = train_label[train_label < 2]
test_label = test_label[test_label < 2]
print('training data size: ')
print(train_data.shape)
print('testing data size: ')
print(test_data.shape)

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, args.source), 'rb') as f:
    scd = pickle.load(f)


eps = args.epsilon

if not os.path.exists('temp_data'):
    os.makedirs('temp_data')

if args.type == 'scd':
    test_data_adv = np.zeros_like(test_data)
    yp = scd.predict(test_data).reshape((-1, 1))
    test_data_adv[:, scd.best_w_index] -= eps * (yp * 2 - 1) * scd.best_w.numpy()
    test_data_adv += test_data
    np.clip(test_data_adv, 0, 1, out=test_data_adv)


elif args.type == 'svm':
    test_data_adv = np.zeros_like(test_data)
    yp = scd.predict(test_data).reshape((-1, 1))
    test_data_adv -= eps * (yp * 2 - 1) * scd.coef_ / np.linalg.norm(scd.coef_)
    test_data_adv += test_data
    np.clip(test_data_adv, 0, 1, out=test_data_adv)
    # yp = scd.predict(test_data)
    # print(accuracy_score(test_label, yp))
    # yp = scd.predict(test_data_adv)
    # print(accuracy_score(test_label, yp))

print('avg l2: %f', np.linalg.norm(test_data_adv - test_data, axis=1).mean())
with open(os.path.join(save_path, args.target), 'rb') as f:
    scd = pickle.load(f)

if args.predict == 'scd':
    yp = scd.predict(test_data_adv, kind='vote')
elif args.predict == 'svm':
    yp = scd.predict(test_data_adv)
print(accuracy_score(test_label, yp))