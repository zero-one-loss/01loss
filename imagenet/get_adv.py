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
args = parser.parse_args()

curdir = os.getcwd()
os.chdir('../data/imagenet')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')
train_data = np.load('train_image.npy').reshape((-1, 224*224*3)) 
test_data = np.load('test_image.npy').reshape((-1, 224*224*3)) 
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
    train_data_adv = np.zeros_like(train_data)
    yp = scd.predict(train_data).reshape((-1, 1))
    train_data_adv[:, scd.best_w_index] -= eps * (yp * 2 - 1) * scd.best_w.numpy()
    train_data_adv += train_data
    np.clip(train_data_adv, 0, 1, out=train_data_adv)

    np.save('temp_data/train_data_adv.npy', train_data_adv)
    print('save adv successfully')

elif args.type == 'svm':
    train_data_adv = np.zeros_like(train_data)
    yp = scd.predict(train_data).reshape((-1, 1))
    train_data_adv -= eps * (yp * 2 - 1) * scd.coef_ / np.linalg.norm(scd.coef_)
    train_data_adv += train_data
    np.clip(train_data_adv, 0, 1, out=train_data_adv)

    np.save('temp_data/train_data_adv_svm.npy', train_data_adv)
    print('save adv successfully')