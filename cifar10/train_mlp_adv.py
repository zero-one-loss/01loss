import sys
sys.path.append('..')
import numpy as np
from core.mlp_converter import MLPadv
import argparse
import os
from sklearn.metrics import accuracy_score
import torch
import pickle

parser = argparse.ArgumentParser(description='SCD 01 loss')

parser.add_argument('--source', default='cifar10_mlp_adv.t7', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--target', default='scd', type=str,
                    help='checkpoint\'s name')

args = parser.parse_args()
print(sys.argv)

curdir = os.getcwd()
os.chdir('../data/cifar10')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')
train_data = np.load('train_image.npy').astype(np.float32).reshape((-1, 32*32*3)) / 255
test_data = np.load('test_image.npy').astype(np.float32).reshape((-1, 32*32*3)) / 255
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

np.random.seed(2018)


state_dict = torch.load(args.source)
w1 = state_dict['linear1.weight'].cpu().numpy().transpose()
b1 = state_dict['linear1.bias'].cpu().numpy().reshape((1, -1))
w2 = state_dict['linear2.weight'].cpu().numpy().transpose()
b2 = state_dict['linear2.bias'].cpu().numpy().reshape((1, -1))

scd = MLPadv(w1, w2, b1, b2)

yp = scd.predict(test)

print('Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Test Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, args.target), 'wb') as f:
    pickle.dump(scd, f)